import sys
sys.path.append('/home/starwit01/workspaces/hanna/movement-predictor')

from movementpredictor.cnn import probabilistic_regression
from movementpredictor.data.datamanagement import getTrackedBaseData
from movementpredictor.data.datafilterer import DataFilterer
from movementpredictor.config import ModelConfig
from movementpredictor.cnn.inferencing import inference_with_prob_calculation
from movementpredictor.clustering import clusterer

import logging
import os
import torch
import matplotlib.pyplot as plt
import json

log = logging.getLogger(__name__)
config = ModelConfig()


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.load(config.path_model, map_location=device) 

    #model = mixture_density_model.MixtureDensityNetwork()
    model = probabilistic_regression.CNN()
    model.load_state_dict(weights, strict=True)
    model.to(device)
    model.eval()

    clusterer.visualValidation(model, config.path_store_data)
    probs, var_size, mus, covs= inference_with_prob_calculation(model, config.path_store_data, "clustering")
    p_thr, v_thr = clusterer.output_distribution(probs, var_size)#, 0.1, 99.9)
    clusterer.plot_unlikely_samples(config.path_store_data, p_thr, v_thr, probs, var_size, mus, covs)
    return
    position_stats, variance_stats = clusterer.output_distribution_pointMap(model, config.path_store_data)
    clusters = clusterer.apply_clustering(position_stats[:-1], variance_stats[:-1])
    #storeParameter(threshold)
    
if __name__ == "__main__":
    main()
