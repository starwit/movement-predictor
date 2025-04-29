import os
import sys
sys.path.append('/home/starwit01/workspaces/hanna/movement-predictor')

from movementpredictor.cnn import model_architectures
from movementpredictor.config import ModelConfig
from movementpredictor.cnn.inferencing import inference_with_stats, visualValidation
from movementpredictor.anomalydetection import clusterer
from movementpredictor.anomalydetection import anomaly_detector
from movementpredictor.data import dataset, datamanagement

import logging
import json

log = logging.getLogger(__name__)
config = ModelConfig()


def main():
    model = model_architectures.get_model(config.model_architecture, config.output_distribution, config.path_model)
    model.eval()

    with open(os.path.join(config.path_model, "parameters.json"), "r") as json_file:
        paras = json.load(json_file)

    ds = dataset.getTorchDataSet(os.path.join(config.path_store_data, "test"), pixel_per_axis=paras["pixel_per_axis"])
    test = dataset.getTorchDataLoader(ds, shuffle=False)

    frame = datamanagement.load_background_frame(config.path_store_data)
    visualValidation(model, test, frame, config.path_plots, num_plots=100)
    #return
    samples_with_stats = inference_with_stats(model, test)

    dist_thr = anomaly_detector.calculate_and_visualize_threshold(samples_with_stats, config.path_plots, config.percentage_anomaly)
    anomaly_detector.store_parameter(config.path_model, dist_thr, config.percentage_anomaly)
    #anomaly_detector.plot_unlikely_samples(samples_with_stats, frame, test, dist_thr, config.path_plots)

    anomalies = anomaly_detector.get_meaningful_unlikely_samples(samples_with_stats, dist_thr)
    anomaly_detector.anomalies_with_video(anomalies, config.path_sae_data_test, config.pixel_per_axis, config.path_plots)

    return
    clustering_vectors, normalization_paras = clusterer.get_clustering_vectors(anomaly_inputs, anomaly_targets, anomaly_mus, anomaly_probs)
    clusters = clusterer.apply_clustering(clustering_vectors)
    clusterer.plot_anomalies_per_cluster(clusters, anomaly_inputs, anomaly_targets, anomaly_mus, anomaly_covs, anomaly_ts, config.path_sae_data, config.dim_x, config.dim_y)
    #storeParameter(p_thr, normalization_paras)
    
if __name__ == "__main__":
    main()
