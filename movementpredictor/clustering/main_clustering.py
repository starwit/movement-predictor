import sys
sys.path.append('/home/starwit01/workspaces/hanna/movement-predictor')

from movementpredictor.cnn import probabilistic_regression
from movementpredictor.config import ModelConfig
from movementpredictor.cnn.inferencing import inference_with_stats
from movementpredictor.clustering import clusterer
from movementpredictor.clustering import anomaly_detector

import logging
import torch

log = logging.getLogger(__name__)
config = ModelConfig()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.load(config.path_model, map_location=device) 

    model = probabilistic_regression.CNN()
    model.load_state_dict(weights, strict=True)
    model.to(device)
    model.eval()

    anomaly_detector.visualValidation(model, config.path_store_data)
    probs, var_size, mus, covs, inps, tars, ad_info = inference_with_stats(model, config.path_store_data, "clustering")
    p_thr, v_thr = anomaly_detector.output_distribution(probs, var_size)#, 0.5)
    #anomaly_detector.plot_unlikely_samples(config.path_store_data, p_thr, v_thr, probs, var_size, mus, covs)
    
    anomaly_inputs, anomaly_targets, anomaly_mus, anomaly_covs, anomaly_probs, anomaly_ts, anomaly_id = anomaly_detector.get_meaningful_unlikely_samples(probs, mus, covs, inps, tars, p_thr, ad_info)
    clustering_vectors, normalization_paras = clusterer.get_clustering_vectors(anomaly_inputs, anomaly_targets, anomaly_mus, anomaly_probs)
    clusters = clusterer.apply_clustering(clustering_vectors)
    clusterer.plot_anomalies_per_cluster(clusters, anomaly_inputs, anomaly_targets, anomaly_mus, anomaly_covs, anomaly_ts, config.path_sae_data, config.dim_x, config.dim_y)
    #storeParameter(p_thr, normalization_paras)
    
if __name__ == "__main__":
    main()
