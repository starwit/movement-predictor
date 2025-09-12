import pytest
from movementpredictor.data.datafilterer import DataFilterer
from unittest.mock import MagicMock

class DummyTrackedObjectPosition:
    def __init__(self, uuid, bbox, center, capture_ts):
        self.uuid = uuid
        self.bbox = bbox
        self.center = center
        self.capture_ts = capture_ts
        self.clear_detection = False
        self.movement_angle = None

@pytest.fixture
def sample_tracks():
    # 2 objects, one moving, one not
    obj1 = [
        DummyTrackedObjectPosition(
            uuid="car1",
            bbox=[[[0, 0], [1, 1]]][0],
            center=[0.0 + i*0.1, 0.0 + i*0.1],
            capture_ts=1000 * i
        ) for i in range(10)
    ]
    obj2 = [
        DummyTrackedObjectPosition(
            uuid="car2",
            bbox=[[[1, 1], [2, 2]]][0],
            center=[1.0, 1.0],
            capture_ts=1000 * i
        ) for i in range(10)
    ]
    return obj1 + obj2

def test_apply_filtering_removes_parking_cars(sample_tracks):
    df = DataFilterer()
    result = df.apply_filtering(sample_tracks, remove_parking_cars=True)
    # Only car1 should remain, as car2 does not move
    assert "car1" in result
    assert "car2" not in result
    assert isinstance(result["car1"], list)
    assert all(isinstance(t, DummyTrackedObjectPosition) for t in result["car1"])

def test_apply_filtering_keeps_all_when_no_removal(sample_tracks):
    df = DataFilterer()
    result = df.apply_filtering(sample_tracks, remove_parking_cars=False)
    # Both car1 and car2 should remain
    assert "car1" in result
    assert "car2" in result
    assert isinstance(result["car1"], list)
    assert isinstance(result["car2"], list)

def test_apply_filtering_empty_input():
    df = DataFilterer()
    result = df.apply_filtering([], remove_parking_cars=True)
    assert result == {}

def test_apply_filtering_min_length_filter():
    df = DataFilterer()
    # Only 3 points, less than min_length (7)
    short_track = [
        DummyTrackedObjectPosition(
            uuid="short",
            bbox=[[[0, 0], [1, 1]]][0],
            center=[0.0 + i*0.1, 0.0 + i*0.1],
            capture_ts=1000 * i
        ) for i in range(3)
    ]
    result = df.apply_filtering(short_track, remove_parking_cars=True)
    assert result == {}