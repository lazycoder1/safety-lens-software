import threading

import pytest

import gstreamer_capture


def test_pipeline_uses_bilinear_scaling_for_detection_quality():
    description = gstreamer_capture._pipeline_description(5_000_000)

    assert "nvvidconv name=converter interpolation-method=1" in description
    assert "decodebin name=decoder_bin" in description
    assert "videorate name=rate_limiter drop-only=false" in description
    assert "capsfilter name=rate_caps" in description
    assert "tcp-timeout=5000000" in description
    assert "emit-signals" not in description


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 2_147_483_647),
        (0, 2_147_483_647),
        (float("nan"), 2_147_483_647),
        (0.2, 1),
        (6, 6),
        (6.1, 7),
        (1_000, 240),
    ],
)
def test_max_rate_is_bounded_and_never_undershoots(value, expected):
    assert gstreamer_capture._bounded_max_rate(value) == expected


def test_rate_caps_force_timestamp_only_streams_to_negotiate_requested_rate():
    assert gstreamer_capture._rate_caps_description(6) == "video/x-raw,framerate=6/1"
    assert (
        gstreamer_capture._rate_caps_description(2_147_483_647)
        == "video/x-raw"
    )


@pytest.mark.parametrize(
    ("width", "height", "maximum", "expected"),
    [
        (1920, 1080, 960, (960, 540)),
        (1080, 1920, 960, (540, 960)),
        (352, 288, 960, (352, 288)),
        (1919, 1080, 960, (960, 540)),
        (1920, 1080, 0, (1920, 1080)),
    ],
)
def test_bounded_dimensions_preserve_orientation_and_aspect(
    width,
    height,
    maximum,
    expected,
):
    assert gstreamer_capture._bounded_dimensions(width, height, maximum) == expected


def test_output_caps_are_fixed_before_frames_arrive():
    assert gstreamer_capture._output_caps_description(1920, 1080, 960) == (
        "video/x-raw,format=BGRx,"
        "width=960,height=540,pixel-aspect-ratio=1/1"
    )
    assert (
        gstreamer_capture._output_caps_description(352, 288, 960)
        == "video/x-raw,format=BGRx"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 0),
        (-1, 0),
        (0, 0),
        (3, 3),
        (31, 30),
        ("invalid", 0),
    ],
)
def test_decoder_drop_interval_is_safely_bounded(value, expected):
    assert gstreamer_capture._bounded_decoder_drop_interval(value) == expected


def test_decoder_drop_interval_only_applies_to_nvdec():
    class Factory:
        def __init__(self, name):
            self._name = name

        def get_name(self):
            return self._name

    class Element:
        def __init__(self, name, supports_property=True):
            self.factory = Factory(name)
            self.supports_property = supports_property
            self.values = {}

        def get_factory(self):
            return self.factory

        def find_property(self, _name):
            return object() if self.supports_property else None

        def set_property(self, name, value):
            self.values[name] = value

    decoder = Element("nvv4l2decoder")
    software_decoder = Element("avdec_h264")

    assert gstreamer_capture._configure_decoder_drop_interval(decoder, 3) is True
    assert decoder.values == {"drop-frame-interval": 3}
    assert (
        gstreamer_capture._configure_decoder_drop_interval(software_decoder, 3)
        is False
    )
    assert software_decoder.values == {}


def test_selected_decoder_factory_controls_hardware_backend_truthfully():
    class Factory:
        def __init__(self, name, klass):
            self.name = name
            self.klass = klass

        def get_name(self):
            return self.name

        def get_metadata(self, key):
            assert key == "klass"
            return self.klass

    class Element:
        def __init__(self, name, klass):
            self.factory = Factory(name, klass)

        def get_factory(self):
            return self.factory

        def find_property(self, _name):
            return None

    capture = object.__new__(gstreamer_capture.GStreamerCapture)
    capture._decoder_drop_interval = 0
    capture._decoder_drop_applied = False
    capture._selected_decoder_factory = None
    capture.capture_backend = "gstreamer_unknown"

    capture._on_decoder_element_added(
        None,
        Element("avdec_h264", "Codec/Decoder/Video"),
    )

    assert capture._selected_decoder_factory == "avdec_h264"
    assert capture.capture_backend == "gstreamer_software"
    assert capture._decoder_drop_applied is False

    capture._on_decoder_element_added(
        None,
        Element("nvv4l2decoder", "Codec/Decoder/Video/Hardware"),
    )
    assert capture._selected_decoder_factory == "nvv4l2decoder"
    assert capture.capture_backend == "gstreamer_nvdec"


def test_appsink_replacements_are_added_when_runtime_exposes_counter():
    class Counters:
        def __init__(self, **values):
            self.values = values

        def get_property(self, name):
            return self.values[name]

    capture = object.__new__(gstreamer_capture.GStreamerCapture)
    capture._rate_limiter = Counters(drop=3, duplicate=2)
    capture._sink = Counters(dropped=5)
    capture._sink_drop_counter_supported = True
    capture._sink_replacement_probe_supported = False
    capture._last_rate_drop_count = 0
    capture._last_rate_duplicate_count = 0
    capture._last_sink_drop_count = 0

    assert capture.consume_capture_policy_counts() == (8, 2)

    capture._rate_limiter.values.update(drop=7, duplicate=3)
    capture._sink.values["dropped"] = 8
    assert capture.consume_capture_policy_counts() == (7, 1)

    # Pipeline renegotiation may reset either cumulative counter.  The new
    # value is counted without ever producing a negative telemetry delta.
    capture._rate_limiter.values.update(drop=1, duplicate=0)
    capture._sink.values["dropped"] = 2
    assert capture.consume_capture_policy_counts() == (3, 0)


def test_older_appsink_runtime_stays_low_overhead_and_reports_lower_bound():
    class Counters:
        def __init__(self, **values):
            self.values = values

        def get_property(self, name):
            if name == "dropped":
                raise AssertionError("unsupported appsink counter must not be read")
            return self.values[name]

    capture = object.__new__(gstreamer_capture.GStreamerCapture)
    capture._rate_limiter = Counters(drop=4, duplicate=1)
    capture._sink = Counters()
    capture._sink_drop_counter_supported = False
    capture._sink_replacement_probe_supported = False
    capture._last_rate_drop_count = 0
    capture._last_rate_duplicate_count = 0
    capture._last_sink_drop_count = 0
    capture._decoder_drop_interval = 0
    capture._decoder_drop_applied = False

    assert capture.consume_capture_policy_counts() == (4, 1)
    assert capture.capture_policy_telemetry() == {
        "appsinkLatestBufferDropsObservable": False,
        "appsinkLatestBufferDropMethod": "unavailable",
        "captureDropAccounting": "videorate-only",
        "captureDropCountIsLowerBound": True,
        "decoderPolicyDropAccounting": "not-configured",
    }


def test_decoder_policy_drop_limit_is_disclosed_as_unobservable():
    capture = object.__new__(gstreamer_capture.GStreamerCapture)
    capture._sink_drop_counter_supported = True
    capture._sink_replacement_probe_supported = False
    capture._decoder_drop_interval = 3
    capture._decoder_drop_applied = True

    assert capture.capture_policy_telemetry() == {
        "appsinkLatestBufferDropsObservable": True,
        "appsinkLatestBufferDropMethod": "native-counter",
        "captureDropAccounting": "videorate-plus-appsink",
        "captureDropCountIsLowerBound": True,
        "decoderPolicyDropAccounting": "configured-not-observable",
    }


def test_sink_pad_probe_counts_latest_buffer_replacements_without_mapping():
    class Counters:
        def get_property(self, name):
            return {"drop": 2, "duplicate": 0}[name]

    capture = object.__new__(gstreamer_capture.GStreamerCapture)
    capture._rate_limiter = Counters()
    capture._sink = None
    capture._sink_drop_counter_supported = False
    capture._sink_replacement_probe_supported = True
    capture._last_rate_drop_count = 0
    capture._last_rate_duplicate_count = 0
    capture._last_sink_drop_count = 0
    capture._sink_probe_lock = threading.Lock()
    capture._sink_probe_arrival_count = 0
    capture._sink_probe_pull_count = 0
    capture._sink_probe_drop_lower_bound = 0
    capture._decoder_drop_interval = 0
    capture._decoder_drop_applied = False

    capture._record_sink_arrival()
    capture._record_sink_arrival()
    capture._record_sink_pull()
    capture._record_sink_arrival()

    assert capture.consume_capture_policy_counts() == (3, 0)
    assert capture.capture_policy_telemetry() == {
        "appsinkLatestBufferDropsObservable": True,
        "appsinkLatestBufferDropMethod": "sink-pad-probe-lower-bound",
        "captureDropAccounting": "videorate-plus-appsink",
        "captureDropCountIsLowerBound": True,
        "decoderPolicyDropAccounting": "not-configured",
    }


def test_padded_bgr_rows_are_copied_without_padding_corruption():
    # Two 2-pixel BGR rows (6 bytes each) in an 8-byte aligned plane.
    padded = bytes(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            250,
            251,
            7,
            8,
            9,
            10,
            11,
            12,
            252,
            253,
        ]
    )

    frame = gstreamer_capture._copy_bgr_plane(
        padded,
        width=2,
        height=2,
        stride=8,
    )

    assert frame is not None
    assert frame.tolist() == [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ]
