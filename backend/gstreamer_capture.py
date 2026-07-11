"""Latest-frame RTSP capture backed by Jetson GStreamer/NVDEC."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
except (ImportError, ValueError):  # pragma: no cover - depends on Jetson image.
    Gst = None


_REQUIRED_ELEMENTS = (
    "rtspsrc",
    "decodebin",
    "nvv4l2decoder",
    "nvvidconv",
    "videoconvert",
    "appsink",
)

_NVVIDCONV_INTERPOLATION_METHOD = 1  # Bilinear; preserves detector confidence at 960px.


def _pipeline_description(tcp_timeout_us: int) -> str:
    return (
        "rtspsrc name=source protocols=tcp latency=100 drop-on-latency=true "
        f"tcp-timeout={tcp_timeout_us} "
        "! decodebin "
        "! nvvidconv name=converter "
        f"interpolation-method={_NVVIDCONV_INTERPOLATION_METHOD} "
        "! capsfilter name=scale_caps "
        "! videoconvert ! video/x-raw,format=BGR "
        "! appsink name=sink sync=false max-buffers=1 drop=true"
    )


def nvdec_runtime_available() -> bool:
    """Return whether the complete in-process NVDEC pipeline is usable."""
    if Gst is None:
        return False
    Gst.init(None)
    return all(Gst.ElementFactory.find(name) is not None for name in _REQUIRED_ELEMENTS)


def _bounded_dimensions(width: int, height: int, max_dimension: int) -> tuple[int, int]:
    """Return an even, aspect-preserving size bounded by max_dimension."""
    width = max(1, int(width))
    height = max(1, int(height))
    maximum = max(0, int(max_dimension))
    if maximum <= 0 or max(width, height) <= maximum:
        return width, height
    scale = maximum / max(width, height)
    target_width = max(2, int(round(width * scale)))
    target_height = max(2, int(round(height * scale)))
    target_width -= target_width % 2
    target_height -= target_height % 2
    return target_width, target_height


def _output_caps_description(width: int, height: int, max_dimension: int) -> str:
    """Return caps fixed from decoded dimensions before the first frame."""
    target_width, target_height = _bounded_dimensions(width, height, max_dimension)
    if (target_width, target_height) == (width, height):
        return "video/x-raw,format=BGRx"
    return (
        "video/x-raw,format=BGRx,"
        f"width={target_width},height={target_height},pixel-aspect-ratio=1/1"
    )


class GStreamerCapture:
    """Small cv2.VideoCapture-compatible wrapper around a latest-frame appsink."""

    delivers_latest_frame = True
    capture_backend = "gstreamer_nvdec"

    def __init__(
        self,
        source: str,
        *,
        open_timeout_ms: int,
        read_timeout_ms: int,
        max_dimension: int = 0,
    ) -> None:
        if not nvdec_runtime_available():
            raise RuntimeError("Jetson NVDEC GStreamer runtime is unavailable")

        self._read_timeout_ns = int(read_timeout_ms) * Gst.MSECOND
        self._opened = False
        self._pending_frame: np.ndarray | None = None
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self._pipeline: Any = None
        self._sink: Any = None
        self._bus: Any = None
        self._scale_filter: Any = None

        tcp_timeout_us = max(1_000_000, int(read_timeout_ms) * 1_000)
        pipeline = Gst.parse_launch(_pipeline_description(tcp_timeout_us))
        source_element = pipeline.get_by_name("source")
        converter = pipeline.get_by_name("converter")
        scale_filter = pipeline.get_by_name("scale_caps")
        sink = pipeline.get_by_name("sink")
        if source_element is None or converter is None or scale_filter is None or sink is None:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("GStreamer RTSP pipeline is incomplete")

        # Set the URL as a property, not in the parse string, so parse errors
        # can never echo credentials embedded in the source.
        source_element.set_property("location", source)
        base_caps = Gst.Caps.from_string("video/x-raw,format=BGRx")
        scale_filter.set_property("caps", base_caps)
        converter_sink = converter.get_static_pad("sink")
        if converter_sink is not None and max_dimension > 0:
            converter_sink.add_probe(
                Gst.PadProbeType.EVENT_DOWNSTREAM,
                self._configure_output_caps,
                (max_dimension, base_caps),
            )
        self._pipeline = pipeline
        self._sink = sink
        self._bus = pipeline.get_bus()
        self._scale_filter = scale_filter
        transition = pipeline.set_state(Gst.State.PLAYING)
        if transition == Gst.StateChangeReturn.FAILURE:
            self.release()
            return

        first_frame = self._pull_frame(int(open_timeout_ms) * Gst.MSECOND)
        if first_frame is None:
            self.release()
            return
        self._pending_frame = first_frame
        self._opened = True

    def _configure_output_caps(self, _pad: Any, info: Any, settings: Any) -> Any:
        """Fix output dimensions from the decoder CAPS event before frame flow."""
        max_dimension, base_caps = settings
        event = info.get_event()
        if event is None or event.type != Gst.EventType.CAPS:
            return Gst.PadProbeReturn.OK
        caps = event.parse_caps()
        if caps is None or caps.get_size() < 1:
            return Gst.PadProbeReturn.OK
        structure = caps.get_structure(0)
        try:
            width = int(structure.get_value("width") or 0)
            height = int(structure.get_value("height") or 0)
        except (TypeError, ValueError):
            return Gst.PadProbeReturn.OK
        if width <= 0 or height <= 0:
            return Gst.PadProbeReturn.OK
        description = _output_caps_description(width, height, max_dimension)
        target_caps = (
            base_caps
            if description == "video/x-raw,format=BGRx"
            else Gst.Caps.from_string(description)
        )
        self._scale_filter.set_property("caps", target_caps)
        return Gst.PadProbeReturn.REMOVE

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._opened:
            return False, None
        if self._pending_frame is not None:
            frame = self._pending_frame
            self._pending_frame = None
            return True, frame

        frame = self._pull_frame(self._read_timeout_ns)
        if frame is None:
            self._opened = False
            return False, None
        return True, frame

    def _pull_frame(self, timeout_ns: int) -> np.ndarray | None:
        if self._sink is None:
            return None
        sample = self._sink.emit("try-pull-sample", max(0, int(timeout_ns)))
        if sample is None:
            self._consume_terminal_message()
            return None

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        if buffer is None or caps is None or caps.get_size() < 1:
            return None
        structure = caps.get_structure(0)
        width = int(structure.get_value("width") or 0)
        height = int(structure.get_value("height") or 0)
        if width <= 0 or height <= 0:
            return None

        ok, mapped = buffer.map(Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            expected = width * height * 3
            if len(mapped.data) < expected:
                return None
            frame = np.frombuffer(mapped.data, dtype=np.uint8, count=expected)
            frame = frame.reshape((height, width, 3)).copy()
        finally:
            buffer.unmap(mapped)

        self._width = width
        self._height = height
        self._fps = self._caps_fps(structure)
        return frame

    @staticmethod
    def _caps_fps(structure: Any) -> float:
        try:
            value = structure.get_value("framerate")
            numerator = float(value.num)
            denominator = float(value.denom)
            return numerator / denominator if denominator else 0.0
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            return 0.0

    def _consume_terminal_message(self) -> None:
        if self._bus is None:
            return
        message = self._bus.timed_pop_filtered(
            0,
            Gst.MessageType.ERROR | Gst.MessageType.EOS,
        )
        if message is not None:
            self._opened = False

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if property_id == cv2.CAP_PROP_FPS:
            return float(self._fps)
        return 0.0

    def set(self, _property_id: int, _value: float) -> bool:
        return False

    def release(self) -> None:
        self._opened = False
        self._pending_frame = None
        pipeline = self._pipeline
        self._pipeline = None
        self._sink = None
        self._bus = None
        self._scale_filter = None
        if pipeline is not None and Gst is not None:
            pipeline.set_state(Gst.State.NULL)
