import logging
import json
import torch
import torch.nn as nn
import os
import sys
import anomalydetection.videogeneration
from aesanomalydetection.recurrentae.ae import LSTM_AE
from aesanomalydetection.recurrentae.validator import plotAnomalTrajectory, plotTrajectory
from aesanomalydetection.recurrentae.dataset import makeTorchPredictionDataSet
from aesanomalydetection.datafilterer import DataFilterer
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


log = logging.getLogger(__name__)


class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr


class Detector():

    def __init__(self, CONFIG):
        log.setLevel(CONFIG.log_level.value)
        self.parameters = Detector.read_json(CONFIG.path_to_model_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTM_AE(self.parameters["dimension_latent_space"]).to(self.device)
        self.whole_video = CONFIG.whole_video

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


    def read_json(file_path):
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
        
    def filter_tracks(self, tracks):
        if len(tracks) != 0:
            tracks = [item for sublist in tracks for item in sublist]
            log.info(f"num tracks before filtering: {len(tracks)}")
            if self.parameters["filtering"]:
                tracks = DataFilterer().apply_filtering(tracks)
                log.info(f"num tracks after filtering: {len(tracks)}")
            else:
                tracks = self._skip_filter_tracks(tracks)
                log.info(f"TESTING WITHOUT FILTERING")
        return tracks
    
    # TESTING WITHOUT FILTERING
    def _skip_filter_tracks(self, tracks):
        mapping = {}
        for track in tracks:
            key = track.uuid
            if key not in mapping:
                mapping[key] = []
            mapping[key].append(track)
        tracks = mapping
        return tracks


    def examine(self, tracks, frames):
        criterion = nn.L1Loss(reduction='sum').to(self.device)
        total_anomalies = []

        for id in tracks.keys():
            anomalies=[] 
            if len(tracks[id]) < 5:
                continue
            with SuppressOutput():      # suppress tqdm progress bar
                trajectories_dataset = makeTorchPredictionDataSet({id: tracks[id]})

            with torch.no_grad():
                for trajectory in trajectories_dataset:
                    trajectory = trajectory.to(self.device).reshape(-1, trajectory.shape[-2], trajectory.shape[-1])
                    target = trajectory
                    pred = self.model(target)
                    loss = criterion(pred, target)
                    if loss.item() > self.parameters["anomaly_loss_threshold"]:
                        log.info("anomaly found")
                        anomalies.append([trajectory, id])

            total_anomalies += anomalies
        
        #TODO move it to anomaly post processing
        self._write_anomalies_to_filesystem(total_anomalies, tracks, frames)
        return total_anomalies
    
    def _write_anomalies_to_filesystem(self, total_anomalies, tracks, frames) :
        if len(total_anomalies) > 0:
            
            with SuppressOutput(): 
                trajectories_dataset = makeTorchPredictionDataSet(tracks)
            for batch in trajectories_dataset:
                plotTrajectory(batch.cpu().numpy(), plotArrows=False)  

            for anomaly in total_anomalies:
                plotAnomalTrajectory(anomaly[0].view(-1, anomaly[0].size(-1)))   
            
            path = "anomalies/anomaly_" + str(datetime.now())
            os.makedirs(path, exist_ok=True)
            plt.savefig(path + "/plot.png")
            plt.close()
            Detector.store_in_json(total_anomalies, path)

            if self.whole_video:
                anomalydetection.videogeneration.storeVideo(frames, path, tracks, total_anomalies, log.level) 
            else:
                anomalydetection.videogeneration.store_frames(frames, path, tracks, total_anomalies, log.level)   

    def store_in_json(anomalies, path):
        new_data = {}
        for _, id in anomalies:
            new_data[str(id)] = [] 
        for anomal_trajectory, id in anomalies:
            new_data[str(id)].append(anomal_trajectory.tolist())

        file_path = path + "/anomal_trajectories.json"

        with open(file_path, 'w') as json_file:
            json.dump(new_data, json_file, indent=4)