from collections import defaultdict
import json
import os
import sys
from typing import List
sys.path.append('/home/starwit01/workspaces/hanna/movement-predictor')

from movementpredictor.cnn import model_architectures
from movementpredictor.cnn.inferencing import inference_with_stats, InferenceResult
from movementpredictor.anomalydetection import anomaly_detector
from movementpredictor.data import dataset
from movementpredictor.evaluation.eval_config import EvalConfig

import logging

log = logging.getLogger(__name__)
evalconfig = EvalConfig()


def store_predictions(predicted_anomalies: List[InferenceResult], path_folder, model_path, num_anomalies, num_anomal_frames_per_trajectory=None, thr_dist=None):
    model_name = make_combined_name(model_path, num_anomal_frames_per_trajectory)
    path_prediction_json = os.path.join(path_folder, model_name + ".json") if thr_dist is not None else os.path.join(path_folder, model_name + "_all_labeled_data.json") 
    os.makedirs(path_folder, exist_ok=True)
    anomaly_dict = defaultdict(list)

    for anomaly in predicted_anomalies:
        anomaly_dict[anomaly.obj_id].append(anomaly)
    
    predictions = defaultdict()
    for obj_id in anomaly_dict.keys():
        timestamps = []
        distances = []
        for sample in anomaly_dict[obj_id]:
            distances.append(sample.prediction.distance_of_target)
            timestamps.append(sample.timestamp)

        predictions[obj_id] = {"distances" : distances, "timestamps": timestamps}

    if thr_dist is not None:
        data = {
            "model_name": model_name,
            "num_anomalies": num_anomalies,
            "threshold_distances": thr_dist,
            "predictions": predictions
        }
    else:
        data = {
            "model_name": model_name,
            "predictions": predictions
        }

    with open(path_prediction_json, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def make_combined_name(path: str, num_frames) -> str:
    folder = os.path.basename(os.path.dirname(path))
    parent_folder = os.path.basename(os.path.dirname(os.path.dirname(path)))
    filename_without_ext = os.path.splitext(os.path.basename(path))[0]
    name = f"{parent_folder}_{folder}_{filename_without_ext}"
    return f"{name}_len{num_frames}" if num_frames is not None else name


def main():
    model = model_architectures.get_model(evalconfig.model_architecture, evalconfig.output_distribution, path_model=evalconfig.path_model)
    model.eval()

    ds = dataset.getTorchDataSet(evalconfig.path_test_data)
    test = dataset.getTorchDataLoader(ds, shuffle=False)

    samples_with_stats = inference_with_stats(model, test)
    print("total test samples: ", len(samples_with_stats))

    for anomaly_length in [1, 2, 5, 10, 20, 50]:
        path_store = os.path.join("movementpredictor/evaluation/plots", evalconfig.camera, "distances_labeling_" + make_combined_name(evalconfig.path_model, anomaly_length))
        dist_thr, anomaly_obj_ids = anomaly_detector.calculate_and_visualize_threshold(samples_with_stats, path_store, 
                                                                                       num_anomalous_trajectories=evalconfig.num_anomalies, num_anomalous_frames_per_id=anomaly_length)
        predicted_anomalies = anomaly_detector.get_unlikely_samples(samples_with_stats, dist_thr, anomaly_obj_ids)
        print("num anomalous tracks: ", len(predicted_anomalies))
        store_predictions(predicted_anomalies, evalconfig.path_store_anomalies, evalconfig.path_model, evalconfig.num_anomalies, anomaly_length, dist_thr)


if __name__ == "__main__":
    main()