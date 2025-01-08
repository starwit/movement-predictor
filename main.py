from movementpredictor.data.datamanagement import getTrackedBaseData
from movementpredictor.data.datafilterer import DataFilterer
from movementpredictor.config import ModelConfig
from movementpredictor.data import dataset
from movementpredictor import validator, mixture_density_model

import logging
import os
import torch
import matplotlib.pyplot as plt
import json
import numpy as np

log = logging.getLogger(__name__)
config = ModelConfig()

def plot_loss_curve(history, path_model):
    directory = os.path.dirname(path_model)
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'], label='Val Loss')
    plt.xlabel('Iteration (in 10000 steps)')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    extended_path = directory + "/plots/loss_curve.png"
    os.makedirs(os.path.dirname(extended_path), exist_ok=True)
    plt.savefig(extended_path)
    plt.show()
    plt.clf()


def storeParameter(threshold):
    parameters = {
        "start_time": config.start_time,
        "end_time": config.end_time,
        "camera_ID": config.camera_id,
        "dimension_latent_space": config.dim_latent,
        "path_to_model": config.path_model,
        "path_to_sae_train_data": config.path_sae_data,
        "anomaly_loss_threshold": threshold,
    }

    directory = os.path.dirname(config.path)
    with open(directory + '/parameters.json', 'w') as json_file:
        json.dump(parameters, json_file, indent=4)

    log.info("Successfully stored model at " + config.path)


def main():

    # TODO: frames in getTrackedBaseData herunterskalieren und speichern, aktuell: im Dataloader, kleinere Bildgröße verwenden
    trackedObjects = getTrackedBaseData(config.path_sae_data)           # 1'220'175 it
    trackedObjects = DataFilterer().apply_filtering(trackedObjects) 

    train, val, test = dataset.makeTorchTrainingDataSets(trackedObjects, config.path_sae_data)
    dataset.plotDataSamples(train, 20)
    return

    if os.path.exists(config.path):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = torch.load(config.path) 

        model = mixture_density_model.MixtureDensityNetwork()
        model.load_state_dict(weights, strict=True)
        model.to(device)
        model.eval()

    else: 
        model, history = mixture_density_model.trainAndStoreAE(train, val, config.path)
        model.eval()
        plot_loss_curve(history, config.path)

    validator.visualValidation(model, test, config.path, background)
    validator.output_distribution(model, test, config.path, background)
    position_stats, variance_stats = validator.output_distribution_pointMap(model, test, config.path, background)
    clusters = validator.apply_clustering(position_stats[:-1], variance_stats[:-1], config.path)
    #storeParameter(threshold)
    
if __name__ == "__main__":
    main()
