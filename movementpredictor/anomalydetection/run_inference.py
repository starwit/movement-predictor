import os
import sys
sys.path.append('/home/starwit01/workspaces/hanna/movement-predictor')

from movementpredictor.cnn import model_architectures
from movementpredictor.config import ModelConfig
from movementpredictor.cnn.inferencing import inference_with_stats, visualValidation
from movementpredictor.anomalydetection import anomaly_detector
from movementpredictor.data import dataset, datamanagement

import logging
import json

log = logging.getLogger(__name__)
config = ModelConfig()


def run_inference_and_calc_anomaly_threshold():
    model = model_architectures.get_model(config.model_architecture, config.output_distribution, config.path_model)
    model.eval()

    with open(os.path.join(config.path_model, "parameters.json"), "r") as json_file:
        paras = json.load(json_file)

    ds = dataset.getTorchDataSet(os.path.join(config.path_store_data, "test"), pixel_per_axis=paras["pixel_per_axis"])
    test = dataset.getTorchDataLoader(ds, shuffle=False)

    if config.visualize:
        frame = datamanagement.load_background_frame(config.path_store_data)
        visualValidation(model, test, frame, config.path_plots, num_plots=100)
    
    samples_with_stats = inference_with_stats(model, test)
    score_thr = anomaly_detector.calculate_trajectory_threshold(samples_with_stats, percentage_p=config.percentage_anomaly)
    anomaly_detector.store_parameter(config.path_model, score_thr, config.percentage_anomaly)

    if config.visualize:
        anomaly_detector.visualize_distances(samples_with_stats, config.path_plots)
        anomalies = anomaly_detector.get_unlikely_trajectories(samples_with_stats, score_thr)
        anomaly_detector.anomalies_with_video(anomalies, config.path_sae_data_test, config.pixel_per_axis, config.path_plots)

    #dist_thr, anomaly_obj_ids = anomaly_detector.calculate_and_visualize_threshold(samples_with_stats, config.path_plots, percentage_p=config.percentage_anomaly)
    #anomalies = anomaly_detector.get_unlikely_samples(samples_with_stats, dist_thr, anomaly_obj_ids)

