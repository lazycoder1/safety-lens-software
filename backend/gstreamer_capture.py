"""Latest-frame RTSP capture backed by Jetson GStreamer/NVDEC."""

from __future__ import annotations

import math
import threading
from typing import Any

import cv2
import numpy as np

try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GstVideo
except (ImportError, ValueError):  # pragma: no cover - depends on Jetson image.
    Gst = None
    GstVideo = None


_REQUIRED_ELEMENTS = (
    "rtspsrc",
    "decodebin",
    "nvv4l2decoder",
    "nvvidconv",
    "videorate",
    "videoconvert",
    "appsink",
)

_NVVIDCONV_INTERPOLATION_METHOD = 1  # Bilinear; preserves detector confidence at 960px.


def _pipeline_description(tcp_timeout_us: int) -> str:
    return (
        "rtspsrc name=source protocols=tcp latency=100 drop-on-latency=true "
        f"tcp-timeout={tcp_timeout_us} "
        "! decodebin name=decoder_bin "
        "! nvvidconv name=converter "
        f"interpolation-method={_NVVIDCONV_INTERPOLATION_METHOD} "
        "! capsfilter name=scale_caps "
        # Allow negotiation for RTSP decoders that expose framerate=0/1.
        # Normal high-rate sources are still dropped to max-rate; a genuinely
        # slower source may duplicate frames so the pipeline remains usable.
        "! videorate name=rate_limiter drop-only=false "
        "! capsfilter name=rate_caps "
        "! videoconvert ! video/x-raw,format=BGR "
        "! appsink name=sink sync=false max-buffers=1 drop=true"
    )


def nvdec_runtime_available() -> bool:
    """Return whether the complete in-process NVDEC pipeline is usable."""
    if Gst is None or GstVideo is None:
        return False
    Gst.init(None)
    return all(Gst.ElementFactory.find(name) is not None for name in _REQUIRED_ELEMENTS)


def _bounded_max_rate(value: float | None) -> int:
    """Return a safe integer GstVideoRate limit or its unlimited sentinel."""
    try:
        parsed = float(value or 0)
    except (TypeError, ValueError):
        return 2_147_483_647
    if not math.isfinite(parsed) or parsed <= 0:
        return 2_147_483_647
    return min(240, max(1, math.ceil(parsed)))


def _rate_caps_description(maximum: int) -> str:
    if maximum >= 2_147_483_647:
        return "video/x-raw"
    return f"video/x-raw,framerate={maximum}/1"


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


def _bounded_decoder_drop_interval(value: int | None) -> int:
    """Return the nvv4l2decoder output interval (zero disables dropping)."""
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return min(30, max(0, parsed))


def _configure_decoder_drop_interval(element: Any, interval: int) -> bool:
    """Apply Jetson decoder output dropping before the pipeline starts."""
    if interval <= 0 or element is None:
        return False
    factory = element.get_factory()
    if factory is None or factory.get_name() != "nvv4l2decoder":
        return False
    if element.find_property("drop-frame-interval") is None:
        return False
    element.set_property("drop-frame-interval", interval)
    return True


def _video_decoder_factory_name(element: Any) -> str | None:
    """Return the selected video decoder factory, ignoring parsers and bins."""
    try:
        factory = element.get_factory()
        if factory is None:
            return None
        name = str(factory.get_name() or "")
        klass = str(factory.get_metadata("klass") or "").lower()
    except (AttributeError, TypeError, ValueError):
        return None
    if not name:
        return None
    if name == "nvv4l2decoder":
        return name
    if "decoder" in klass and "video" in klass:
        return name
    # Some older vendor factories omit klass metadata.  These well-known
    # prefixes still identify actual decoder elements rather than decodebin.
    lowered = name.lower()
    if lowered.startswith("avdec_") or (
        lowered.startswith("v4l2") and lowered.endswith("dec")
    ):
        return name
    return None


def _has_readable_property(element: Any, name: str) -> bool:
    try:
        return element is not None and element.find_property(name) is not None
    except (AttributeError, TypeError, ValueError):
        return False


def _read_nonnegative_counter(element: Any, name: str) -> int | None:
    try:
        value = int(element.get_property(name))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    return max(0, value)


def _cumulative_delta(current: int | None, previous: int) -> tuple[int, int]:
    """Convert a possibly-reset cumulative counter to a monotonic delta."""
    if current is None:
        return 0, previous
    delta = current - previous if current >= previous else current
    return max(0, delta), current


def _copy_bgr_plane(
    data: Any,
    *,
    width: int,
    height: int,
    stride: int,
    offset: int = 0,
) -> np.ndarray | None:
    """Copy one BGR plane row-by-row without treating padding as pixels."""
    row_bytes = int(width) * 3
    rows = int(height)
    plane_stride = int(stride)
    plane_offset = int(offset)
    if row_bytes <= 0 or rows <= 0 or abs(plane_stride) < row_bytes:
        return None
    try:
        available = len(data)
    except TypeError:
        return None
    frame = np.empty((rows, int(width), 3), dtype=np.uint8)
    flat = frame.reshape(rows, row_bytes)
    for row_index in range(rows):
        start = plane_offset + row_index * plane_stride
        end = start + row_bytes
        if start < 0 or end > available:
            return None
        try:
            flat[row_index, :] = np.frombuffer(
                data,
                dtype=np.uint8,
                count=row_bytes,
                offset=start,
            )
        except (TypeError, ValueError):
            return None
    return frame


def _bgr_plane_layout(buffer: Any, caps: Any) -> tuple[int, int] | None:
    """Return negotiated BGR plane stride/offset, honoring buffer metadata."""
    if GstVideo is None:
        return None
    try:
        video_meta = GstVideo.buffer_get_video_meta(buffer)
    except (AttributeError, TypeError, ValueError):
        video_meta = None
    if video_meta is not None:
        try:
            return int(video_meta.stride[0]), int(video_meta.offset[0])
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
    try:
        parsed, info = GstVideo.video_info_from_caps(caps)
    except (AttributeError, TypeError, ValueError):
        # Older PyGObject builds expose the instance method instead of the
        # introspection-friendly out-parameter helper.
        try:
            info = GstVideo.VideoInfo.new()
            parsed = info.from_caps(caps)
        except (AttributeError, TypeError, ValueError):
            return None
    if not parsed:
        return None
    try:
        return int(info.stride[0]), int(info.offset[0])
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


class GStreamerCapture:
    """Small cv2.VideoCapture-compatible wrapper around a latest-frame appsink."""

    delivers_latest_frame = True
    capture_backend = "gstreamer_unknown"

    def __init__(
        self,
        source: str,
        *,
        open_timeout_ms: int,
        read_timeout_ms: int,
        max_dimension: int = 0,
        max_fps: float | None = None,
        decoder_drop_interval: int = 0,
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
        self._rate_limiter: Any = None
        self._rate_filter: Any = None
        self._max_rate = _bounded_max_rate(max_fps)
        self._decoder_drop_interval = _bounded_decoder_drop_interval(
            decoder_drop_interval
        )
        self._decoder_drop_applied = False
        self._selected_decoder_factory: str | None = None
        self.capture_backend = "gstreamer_unknown"
        self._last_rate_drop_count = 0
        self._last_rate_duplicate_count = 0
        self._last_sink_drop_count = 0
        self._sink_probe_lock = threading.Lock()
        self._sink_probe_arrival_count = 0
        self._sink_probe_pull_count = 0
        self._sink_probe_drop_lower_bound = 0
        self._sink_replacement_probe_supported = False
        self._sink_probe_pad: Any = None
        self._sink_probe_id: int | None = None

        tcp_timeout_us = max(1_000_000, int(read_timeout_ms) * 1_000)
        pipeline = Gst.parse_launch(_pipeline_description(tcp_timeout_us))
        source_element = pipeline.get_by_name("source")
        decoder_bin = pipeline.get_by_name("decoder_bin")
        converter = pipeline.get_by_name("converter")
        scale_filter = pipeline.get_by_name("scale_caps")
        rate_limiter = pipeline.get_by_name("rate_limiter")
        rate_filter = pipeline.get_by_name("rate_caps")
        sink = pipeline.get_by_name("sink")
        if (
            source_element is None
            or decoder_bin is None
            or converter is None
            or scale_filter is None
            or rate_limiter is None
            or rate_filter is None
            or sink is None
        ):
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("GStreamer RTSP pipeline is incomplete")

        # Set the URL as a property, not in the parse string, so parse errors
        # can never echo credentials embedded in the source.
        source_element.set_property("location", source)
        # Always inspect decodebin's selected child.  Merely having the NVDEC
        # plugin installed does not prove it won autoplug selection.
        decoder_bin.connect("element-added", self._on_decoder_element_added)
        rate_limiter.set_property("max-rate", self._max_rate)
        rate_filter.set_property(
            "caps",
            Gst.Caps.from_string(_rate_caps_description(self._max_rate)),
        )
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
        # GStreamer 1.28 added a cheap cumulative appsink `dropped` property.
        # JetPack 5's older runtime does not have it.  Poll when present and do
        # not enable emit-signals: those callbacks add per-frame overhead.
        self._sink_drop_counter_supported = _has_readable_property(sink, "dropped")
        if not self._sink_drop_counter_supported:
            self._install_sink_replacement_probe(sink)
        self._bus = pipeline.get_bus()
        self._scale_filter = scale_filter
        self._rate_limiter = rate_limiter
        self._rate_filter = rate_filter
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

    def _on_decoder_element_added(
        self,
        _container: Any,
        element: Any,
    ) -> None:
        """Record the decoder actually selected and configure NVDEC if present."""
        factory_name = _video_decoder_factory_name(element)
        if factory_name is None:
            return
        self._selected_decoder_factory = factory_name
        self.capture_backend = (
            "gstreamer_nvdec"
            if factory_name == "nvv4l2decoder"
            else "gstreamer_software"
        )
        if _configure_decoder_drop_interval(
            element,
            self._decoder_drop_interval,
        ):
            self._decoder_drop_applied = True

    def _install_sink_replacement_probe(self, sink: Any) -> None:
        """Observe post-videorate arrivals without mapping or copying frames."""
        try:
            pad = sink.get_static_pad("sink")
            if pad is None:
                return
            probe_id = pad.add_probe(
                Gst.PadProbeType.BUFFER,
                self._on_sink_buffer,
            )
        except (AttributeError, TypeError, ValueError):
            return
        if not probe_id:
            return
        self._sink_probe_pad = pad
        self._sink_probe_id = int(probe_id)
        self._sink_replacement_probe_supported = True

    def _record_sink_arrival(self) -> None:
        with self._sink_probe_lock:
            self._sink_probe_arrival_count += 1
            self._update_sink_probe_drop_lower_bound()

    def _record_sink_pull(self) -> None:
        with self._sink_probe_lock:
            self._sink_probe_pull_count += 1
            self._update_sink_probe_drop_lower_bound()

    def _update_sink_probe_drop_lower_bound(self) -> None:
        # appsink holds at most one buffer.  Subtracting one possible queued
        # buffer avoids a pull/arrival race and undercounts by at most one over
        # the capture lifetime.  Keep the cumulative bound monotonic after a
        # pull empties the queue.
        candidate = max(
            0,
            self._sink_probe_arrival_count - self._sink_probe_pull_count - 1,
        )
        self._sink_probe_drop_lower_bound = max(
            self._sink_probe_drop_lower_bound,
            candidate,
        )

    def _on_sink_buffer(self, _pad: Any, _info: Any) -> Any:
        self._record_sink_arrival()
        return Gst.PadProbeReturn.OK

    def _sink_probe_drops(self) -> int:
        with self._sink_probe_lock:
            return self._sink_probe_drop_lower_bound

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
        if self._sink_replacement_probe_supported:
            self._record_sink_pull()

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
            layout = _bgr_plane_layout(buffer, caps)
            if layout is None:
                return None
            stride, offset = layout
            frame = _copy_bgr_plane(
                mapped.data,
                width=width,
                height=height,
                stride=stride,
                offset=offset,
            )
            if frame is None:
                return None
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

    def set_max_fps(self, value: float) -> None:
        """Update the pre-conversion drop rate after a live config change."""
        maximum = _bounded_max_rate(value)
        if maximum == self._max_rate:
            return
        if self._rate_limiter is not None:
            self._rate_limiter.set_property("max-rate", maximum)
        if self._rate_filter is not None:
            self._rate_filter.set_property(
                "caps",
                Gst.Caps.from_string(_rate_caps_description(maximum)),
            )
        self._max_rate = maximum

    def consume_capture_policy_counts(self) -> tuple[int, int]:
        """Return observable policy drops/duplicates since the previous read.

        GStreamer's counters are cumulative and may reset when the pipeline is
        renegotiated.  Converting them to non-negative deltas keeps the public
        telemetry monotonic without exposing any stream metadata.  Recent
        runtimes also expose appsink latest-buffer replacements; older JetPack
        versions do not, so their capture drop count remains a documented lower
        bound rather than enabling expensive signal callbacks.
        """
        limiter = self._rate_limiter
        rate_dropped = (
            _read_nonnegative_counter(limiter, "drop")
            if limiter is not None
            else None
        )
        rate_duplicated = (
            _read_nonnegative_counter(limiter, "duplicate")
            if limiter is not None
            else None
        )
        rate_drop_delta, self._last_rate_drop_count = _cumulative_delta(
            rate_dropped,
            self._last_rate_drop_count,
        )
        duplicate_delta, self._last_rate_duplicate_count = _cumulative_delta(
            rate_duplicated,
            self._last_rate_duplicate_count,
        )
        sink_dropped = (
            _read_nonnegative_counter(self._sink, "dropped")
            if self._sink_drop_counter_supported
            else (
                self._sink_probe_drops()
                if self._sink_replacement_probe_supported
                else None
            )
        )
        sink_drop_delta, self._last_sink_drop_count = _cumulative_delta(
            sink_dropped,
            self._last_sink_drop_count,
        )
        return rate_drop_delta + sink_drop_delta, duplicate_delta

    def capture_policy_telemetry(self) -> dict[str, Any]:
        """Describe exactly which configured drop stages are observable."""
        appsink_drops_observable = (
            self._sink_drop_counter_supported
            or self._sink_replacement_probe_supported
        )
        if self._sink_drop_counter_supported:
            appsink_drop_method = "native-counter"
        elif self._sink_replacement_probe_supported:
            appsink_drop_method = "sink-pad-probe-lower-bound"
        else:
            appsink_drop_method = "unavailable"
        if self._decoder_drop_interval <= 0:
            decoder_accounting = "not-configured"
        elif self._decoder_drop_applied:
            decoder_accounting = "configured-not-observable"
        else:
            decoder_accounting = "requested-not-applied"
        return {
            "appsinkLatestBufferDropsObservable": appsink_drops_observable,
            "appsinkLatestBufferDropMethod": appsink_drop_method,
            "captureDropAccounting": (
                "videorate-plus-appsink"
                if appsink_drops_observable
                else "videorate-only"
            ),
            "captureDropCountIsLowerBound": (
                not self._sink_drop_counter_supported
                or self._decoder_drop_applied
            ),
            "decoderPolicyDropAccounting": decoder_accounting,
        }

    def release(self) -> None:
        self._opened = False
        self._pending_frame = None
        probe_pad = self._sink_probe_pad
        probe_id = self._sink_probe_id
        self._sink_probe_pad = None
        self._sink_probe_id = None
        self._sink_replacement_probe_supported = False
        if probe_pad is not None and probe_id is not None:
            try:
                probe_pad.remove_probe(probe_id)
            except (AttributeError, TypeError, ValueError):
                pass
        pipeline = self._pipeline
        self._pipeline = None
        self._sink = None
        self._bus = None
        self._scale_filter = None
        self._rate_limiter = None
        self._rate_filter = None
        if pipeline is not None and Gst is not None:
            pipeline.set_state(Gst.State.NULL)
