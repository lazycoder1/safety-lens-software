import pytest

import gstreamer_capture


def test_pipeline_uses_bilinear_scaling_for_detection_quality():
    description = gstreamer_capture._pipeline_description(5_000_000)

    assert "nvvidconv name=converter interpolation-method=1" in description
    assert "decodebin name=decoder_bin" in description
    assert "videorate name=rate_limiter drop-only=false" in description
    assert "capsfilter name=rate_caps" in description
    assert "tcp-timeout=5000000" in description


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
