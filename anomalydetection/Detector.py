import logging
import json
import torch
import torch.nn as nn
import os
import time
from AEsAnomalyDetection.RecurrentAE.AE import LSTM_AE
from AEsAnomalyDetection.RecurrentAE.Validator import plotAnomalTrajectory, plotTrajectory
from AEsAnomalyDetection.RecurrentAE.Dataset import makeTorchPredictionDataSet
from AEsAnomalyDetection.DataFilterer import DataFilterer
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


log = logging.getLogger(__name__)


class Detector():

    def __init__(self, pathModelParameters):
        self.parameters = self.read_json(pathModelParameters)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # import
        self.model = LSTM_AE(self.parameters["dimension_latent_space"]).to(self.device)

        try:
            weights = torch.load(self.parameters["path_to_model"], map_location=torch.device(self.device))
        except:
            log.error(f"Model weights not found at {self.parameters["path_to_model"]}")
            exit(1)
        try:    
            self.model.load_state_dict(weights)
        except:
            log.error(f"Model weights do not fit model architecture LSTM_AE")
            exit(1)

        log.info("starting anomaly detection with parameters:")
        for key in self.parameters.keyset():
            log.info(key + ": " + str(self.parameters[key]))


    def read_json(self, file_path):
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                log.error(f"Could not find model parameters at {file_path}")
                return None
            with file_path.open('r') as file:
                return json.load(file)
        except IOError as e:
            log.error(f"Could not read model parameters {file_path}")
            log.debug(e)
            return None
        
    
    def examine(self, tracks, frames):
        tracks = DataFilterer().apply_filtering(tracks)  
        trajectories_dataset = makeTorchPredictionDataSet(tracks) 
        anomalies = []
        criterion = nn.L1Loss(reduction='sum').to(self.device)

        with torch.no_grad():
            for trajectory in trajectories_dataset:
                trajectory = trajectory.to(self.device).unsqueeze(0)
                target = trajectory
                pred = self.model(target)
                loss = criterion(pred, target)
                if loss.item() > self.parameters["anomaly_loss_threshold"]:
                    anomalies.append(trajectory)

        for anomaly in anomalies:
            for num, batch in enumerate(trajectories_dataset):
                plotTrajectory(batch.cpu().numpy(), plotArrows=False)  
                if num > 5000:
                    break

            plotAnomalTrajectory(anomaly)   
            path = "/anomalies/anomaly_" + str(datetime.now())
            os.makedirs(path, exist_ok=True)
            plt.savefig(path + "/plot.png")
            plt.close()
            storeVideo(anomaly, frames, path) #TODO: implement
        
        return anomalies
    
