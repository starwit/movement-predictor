import pytest
import numpy as np
from collections import namedtuple
from movementpredictor.anomalydetection.anomaly_detector import calculate_trajectory_anomaly_scores
from movementpredictor.anomalydetection.anomaly_detector import calculate_trajectory_threshold
from movementpredictor.anomalydetection.anomaly_detector import calculate_threshold

# Create a mock InferenceResult and Prediction
MockPrediction = namedtuple("MockPrediction", ["distance_of_target"])
MockInferenceResult = namedtuple("MockInferenceResult", ["obj_id", "prediction"])

def make_sample(obj_id, distance):
    return MockInferenceResult(obj_id=obj_id, prediction=MockPrediction(distance_of_target=distance))

def test_single_trajectory_multiple_distances():
    samples = [
        make_sample("id1", 1.0),
        make_sample("id1", 2.0),
        make_sample("id1", 3.0)
    ]
    scores = calculate_trajectory_anomaly_scores(samples, a=1.0)
    assert len(scores) == 1
    obj_id, score = scores[0]
    assert obj_id == "id1"
    # With a=1.0, it's just the mean
    assert np.isclose(score, 2.0)

def test_multiple_trajectories():
    samples = [
        make_sample("id1", 1.0),
        make_sample("id1", 2.0),
        make_sample("id2", 10.0),
        make_sample("id2", 20.0)
    ]
    scores = dict(calculate_trajectory_anomaly_scores(samples, a=1.0))
    assert set(scores.keys()) == {"id1", "id2"}
    assert np.isclose(scores["id1"], 1.5)
    assert np.isclose(scores["id2"], 15.0)

def test_exponential_weighting():
    samples = [
        make_sample("id1", 10.0),
        make_sample("id1", 1.0)
    ]
    # With a=0.5: weights are 1, 0.5; so score = (10*1 + 1*0.5)/(1+0.5) = (10+0.5)/1.5 = 7.0
    scores = calculate_trajectory_anomaly_scores(samples, a=0.5)
    assert len(scores) == 1
    obj_id, score = scores[0]
    assert np.isclose(score, 7.0)

def test_empty_samples():
    scores = calculate_trajectory_anomaly_scores([])
    assert scores == []

def test_single_sample():
    samples = [make_sample("id1", 42.0)]
    scores = calculate_trajectory_anomaly_scores(samples)
    assert len(scores) == 1
    obj_id, score = scores[0]
    assert obj_id == "id1"
    assert np.isclose(score, 42.0)
    
def test_calculate_trajectory_threshold_percentage():
    samples = [
        make_sample("id1", 1.0),
        make_sample("id1", 2.0),
        make_sample("id2", 10.0),
        make_sample("id2", 20.0),
        make_sample("id3", 5.0),
        make_sample("id3", 6.0),
    ]
    # With 33% anomalous (percentage_p=67), 1 out of 3 trajectories should be anomalous
    threshold = calculate_trajectory_threshold(samples, percentage_p=67)
    assert np.isclose(threshold, 15.1, atol=1e-1)

def test_calculate_trajectory_threshold_both_args_warns(monkeypatch):
    samples = [
        make_sample("id1", 1.0),
        make_sample("id2", 2.0),
    ]
    # Should use percentage_p if both are given
    threshold = calculate_trajectory_threshold(samples, percentage_p=50, num_anomalous_trajectories=1)
    # 50% anomalous: 1 out of 2, so threshold is the score of the first
    assert np.isclose(threshold, 2.0)

def test_calculate_trajectory_threshold_error_on_none(monkeypatch):
    samples = [
        make_sample("id1", 1.0),
        make_sample("id2", 2.0),
    ]
    # Should exit(1) if both args are None
    with pytest.raises(SystemExit):
        calculate_trajectory_threshold(samples)

def test_calculate_trajectory_threshold_error_too_many_anomalous(monkeypatch):
    samples = [
        make_sample("id1", 1.0),
        make_sample("id2", 2.0),
    ]
    # 3 anomalous but only 2 trajectories
    with pytest.raises(SystemExit):
        calculate_trajectory_threshold(samples, num_anomalous_trajectories=3)

def test_calculate_threshold_percentage():
    samples = [
        make_sample("id1", 1.0),
        make_sample("id1", 2.0),
        make_sample("id1", 3.0),
        make_sample("id2", 10.0),
        make_sample("id2", 11.0),
        make_sample("id2", 12.0),
        make_sample("id3", 5.0),
        make_sample("id3", 6.0),
        make_sample("id3", 7.0),
    ]
    # 33% anomalous (percentage_p=67), so 1 out of 3 obj_ids
    threshold, anomaly_obj_ids = calculate_threshold(samples, percentage_p=67, num_anomalous_frames_per_id=3)
    # id2 has highest distances, so threshold should be 10.0 (lowest of id2's top 3)
    assert np.isclose(threshold, 10.0)
    assert anomaly_obj_ids == {"id2"}

def test_calculate_threshold_num_anomalous_trajectories():
    samples = [
        make_sample("id1", 1.0),
        make_sample("id1", 2.0),
        make_sample("id1", 3.0),
        make_sample("id2", 10.0),
        make_sample("id2", 11.0),
        make_sample("id2", 12.0),
        make_sample("id3", 5.0),
        make_sample("id3", 6.0),
        make_sample("id3", 7.0),
    ]
    # Ask for 2 anomalous trajectories
    threshold, anomaly_obj_ids = calculate_threshold(samples, num_anomalous_trajectories=2, num_anomalous_frames_per_id=3)
    assert np.isclose(threshold, 5.0)
    assert anomaly_obj_ids == {"id2", "id3"}

def test_calculate_threshold_min_anomalous_frames():
    samples = [
        make_sample("id1", 1.0),
        make_sample("id1", 2.0),
        make_sample("id2", 10.0),
        make_sample("id2", 11.0),
        make_sample("id3", 5.0),
        make_sample("id3", 6.0),
        make_sample("id3", 7.0),
    ]
    # num_anomalous_frames_per_id=2, so id1 is not anomalous (only 2 samples, but threshold will be for 1 trajectory)
    threshold, anomaly_obj_ids = calculate_threshold(samples, num_anomalous_trajectories=1, num_anomalous_frames_per_id=3)
    assert anomaly_obj_ids == {"id3"}
    assert np.isclose(threshold, 5.0)

def test_calculate_threshold_error_on_none():
    samples = [
        make_sample("id1", 1.0),
        make_sample("id2", 2.0),
    ]
    with pytest.raises(SystemExit):
        calculate_threshold(samples)

def test_calculate_threshold_error_too_many_anomalous():
    samples = [
        make_sample("id1", 1.0),
        make_sample("id2", 2.0),
    ]
    with pytest.raises(SystemExit):
        calculate_threshold(samples, num_anomalous_trajectories=3)

def test_calculate_threshold_both_args_warns(monkeypatch):
    samples = [
        make_sample("id1", 1.0),
        make_sample("id1", 2.0),
        make_sample("id2", 10.0),
        make_sample("id2", 20.0),
    ]
    # Should use percentage_p if both are given
    threshold, anomaly_obj_ids = calculate_threshold(samples, percentage_p=50, num_anomalous_trajectories=1, num_anomalous_frames_per_id=2)
    # 50% anomalous: 1 out of 2, so threshold is the lowest of id2's top 2
    assert np.isclose(threshold, 10.0)
    assert anomaly_obj_ids == {"id2"}

