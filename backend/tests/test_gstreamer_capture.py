import pytest

import gstreamer_capture


def test_pipeline_uses_bilinear_scaling_for_detection_quality():
    description = gstreamer_capture._pipeline_description(5_000_000)

    assert "nvvidconv interpolation-method=1" in description
    assert "tcp-timeout=5000000" in description


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
