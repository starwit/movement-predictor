from collections import defaultdict
import json
import os
import sys
from typing import List
sys.path.append('/home/starwit01/workspaces/hanna/movement-predictor')

from movementpredictor.cnn import inferencing, model_architectures
from movementpredictor.cnn.inferencing import inference_with_stats
from movementpredictor.anomalydetection import anomaly_detector
from movementpredictor.data import dataset
from movementpredictor.evaluation.eval_config import EvalConfig

import logging

log = logging.getLogger(__name__)
evalconfig = EvalConfig()


def store_predictions(predicted_anomalies: List[inferencing.InferenceResult], path_folder, model_path, num_anomalies, thr_dist):
    model_name = make_combined_name(model_path)
    path_prediction_json = os.path.join(path_folder, model_name + ".json")
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

    data = {
        "model_name": model_name,
        "num_anomalies": num_anomalies,
        "threshold_distances": thr_dist,
        "predictions": predictions
    }

    with open(path_prediction_json, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def make_combined_name(path: str) -> str:
    folder = os.path.basename(os.path.dirname(path))
    parent_folder = os.path.basename(os.path.dirname(os.path.dirname(path)))
    filename_without_ext = os.path.splitext(os.path.basename(path))[0]
    return f"{parent_folder}_{folder}_{filename_without_ext}"


def main():
    model = model_architectures.get_model(evalconfig.model_architecture, evalconfig.output_distribution, path_model=evalconfig.path_model)
    model.eval()

    ds = dataset.getTorchDataSet(evalconfig.path_test_data)
    test = dataset.getTorchDataLoader(ds, shuffle=False)

    samples_with_stats = inference_with_stats(model, test)
    print("total test samples: ", len(samples_with_stats))

    path_store = os.path.join("movementpredictor/evaluation/plots", evalconfig.camera, "distances_labeling_" + make_combined_name(evalconfig.path_model))
    dist_thr = anomaly_detector.calculate_and_visualize_threshold(samples_with_stats, path_store, num_anomalous_trajectories=evalconfig.num_anomalies)

    predicted_anomalies = anomaly_detector.get_unlikely_samples(samples_with_stats, dist_thr)
    print("num anomalous tracks: ", len(predicted_anomalies))
    store_predictions(predicted_anomalies, evalconfig.path_store_anomalies, evalconfig.path_model, evalconfig.num_anomalies, dist_thr)


if __name__ == "__main__":
    main()