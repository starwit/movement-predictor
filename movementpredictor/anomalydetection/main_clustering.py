import sys
sys.path.append('/home/starwit01/workspaces/hanna/movement-predictor')

from movementpredictor.cnn import probabilistic_regression
from movementpredictor.config import ModelConfig
from movementpredictor.cnn.inferencing import inference_with_stats
from movementpredictor.anomalydetection import clusterer
from movementpredictor.anomalydetection import anomaly_detector
from movementpredictor.data import dataset

import logging
import torch

log = logging.getLogger(__name__)
config = ModelConfig()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.load(config.path_model + "/model_weights.pth", map_location=device) 

    model = probabilistic_regression.CNN()
    model.load_state_dict(weights, strict=True)
    model.to(device)
    model.eval()

    ds = dataset.merge_datasets(config.path_store_data, "clustering")
    test = dataset.getTorchDataLoader(ds, shuffle=False)

    anomaly_detector.visualValidation(model, test)
    #return
    samples_with_stats = inference_with_stats(model, test)
    dist_thr = anomaly_detector.calculate_and_visualize_threshold(samples_with_stats, config.percentage_anomaly)
    anomaly_detector.store_parameter(config.path_model, dist_thr, config.percentage_anomaly)
    anomaly_detector.plot_unlikely_samples(test, dist_thr, samples_with_stats) 

    anomalies = anomaly_detector.get_meaningful_unlikely_samples(samples_with_stats, dist_thr)
    anomaly_detector.anomalies_with_video(anomalies, config.path_sae_data, config.dim_x, config.dim_y)

    return
    clustering_vectors, normalization_paras = clusterer.get_clustering_vectors(anomaly_inputs, anomaly_targets, anomaly_mus, anomaly_probs)
    clusters = clusterer.apply_clustering(clustering_vectors)
    clusterer.plot_anomalies_per_cluster(clusters, anomaly_inputs, anomaly_targets, anomaly_mus, anomaly_covs, anomaly_ts, config.path_sae_data, config.dim_x, config.dim_y)
    #storeParameter(p_thr, normalization_paras)
    
if __name__ == "__main__":
    main()
