import pytest

import gstreamer_capture


def test_pipeline_uses_bilinear_scaling_for_detection_quality():
    description = gstreamer_capture._pipeline_description(5_000_000)

    assert "nvvidconv name=converter interpolation-method=1" in description
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
