import pytest

import movementpredictor.data.data_prep
from movementpredictor.data.datafilterer import DataFilterer


@pytest.mark.parametrize(
    "dx, dy, expected", [
        # dx>0, dy>0 => 45°
        (1.0, 1.0, 45.0),
        # dx<0, dy<0 => 225°
        (-1.0, -1.0, 225.0),
        # dx<0, dy>0 => 135°
        (-1.0, 1.0, 135.0),
        # dx>0, dy<0 => 315°
        (1.0, -1.0, 315.0),
    ]
)


def test_quadrant_angles(dx, dy, expected):
    # To match implementation, center y is set to invert dy
    center = (dx, -dy)
    previous = (0.0, 0.0)

    angle = DataFilterer._get_angle(center, previous)

    assert angle == pytest.approx(expected, abs=1e-6)
    assert angle    == pytest.approx(expected, abs=1e-6)