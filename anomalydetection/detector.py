import logging
import json
import torch
import torch.nn as nn
import os
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
        self.model = LSTM_AE(self.parameters["dimension_latent_space"]).to(self.device)

        try:
            weights = torch.load(self.parameters["path_to_model"], map_location=torch.device(self.device))
        except:
            log.error(f"Model weights not found at {self.parameters['path_to_model']}")
            exit(1)
        try:    
            self.model.load_state_dict(weights)
        except:
            log.error(f"Model weights do not fit model architecture LSTM_AE")
            exit(1)

        log.info("starting anomaly detection with parameters:")
        for key in self.parameters.keys():
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
        
    
    def examine(self, tracks, frames=None):
        tracks = [item for sublist in tracks for item in sublist] # flatten tracks list
        tracks = DataFilterer().apply_filtering(tracks)  
        criterion = nn.L1Loss(reduction='sum').to(self.device)
        total_anomalies = []

        for id in tracks.keys():
            anomalies=[] 
            trajectories_dataset = makeTorchPredictionDataSet({id: tracks[id]})

            with torch.no_grad():
                for trajectory in zip(trajectories_dataset):
                    trajectory = trajectory.to(self.device).unsqueeze(0)
                    target = trajectory
                    pred = self.model(target)
                    loss = criterion(pred, target)
                    if loss.item() > self.parameters["anomaly_loss_threshold"]:
                        anomalies.append([trajectory, id])

            self.store_in_json(anomalies)

            for batch in trajectories_dataset:
                plotTrajectory(batch.cpu().numpy(), plotArrows=False)  

        for anomaly in total_anomalies:
            plotAnomalTrajectory(anomaly[0])   
            
        path = "home/hanna/workspaces/sae-anomaly-detection/anomalies/anomaly_" + str(datetime.now())
        os.makedirs(path, exist_ok=True)
        plt.savefig(path + "/plot.png")
        plt.close()

        #storeVideo(total_anomalies, frames, path) #TODO: implement
        
        return total_anomalies
    

    def store_in_json(anomalies):
        new_data = {}
        for anomal_trajectory, id in anomalies:
            new_data[id] = anomal_trajectory

        file_path = "/anomalies/anomalies.json"

        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
        else:
            data = {}

        data.update(new_data)

        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)