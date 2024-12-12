import logging
import os
import sys

import torch
import torch.nn as nn
from aesanomalydetection.datafilterer import DataFilterer
from aesanomalydetection.recurrentae.ae import LSTM_AE
from aesanomalydetection.recurrentae.dataset import makeTorchPredictionDataSet
from visionapi.anomaly_pb2 import AnomalyMessage, Point, Trajectory

from anomalydetection.config import AnomalyDetectionConfig
from anomalydetection.modelinfocollector import ModelInfoCollector

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

    def __init__(self, CONFIG: AnomalyDetectionConfig) -> None:
        log.setLevel(CONFIG.log_level.value)
        self._config = CONFIG
        self._parameters = ModelInfoCollector(CONFIG).model_parameters
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = LSTM_AE(self._parameters["dimension_latent_space"]).to(self._device)

        try:
            weights = torch.load(self._config.model.weights_path, map_location=torch.device(self._device))
        except:
            log.error(f"Model weights not found at {self._config.model.weights_path}")
            exit(1)
        try:    
            self._model.load_state_dict(weights)
        except:
            log.error(f"Model weights do not fit model architecture LSTM_AE")
            exit(1)

        log.info("starting anomaly detection with parameters:")
        for key in self._parameters.keys():
            log.info(key + ": " + str(self._parameters[key]))
      
    def filter_tracks(self, tracks):
        if len(tracks) != 0:
            tracks = [item for sublist in tracks for item in sublist]
            log.info(f"num tracks before filtering: {len(tracks)}")
            if self._config.filtering:
                tracks = DataFilterer().apply_filtering(tracks)
                log.info(f"num tracks after filtering: {len(tracks)}")
            else:
                tracks = self._skip_filter_tracks(tracks)
                log.info(f"TESTING WITHOUT FILTERING")
        return tracks
    
    def examine_tracks_for_anomalies(self, tracks, frames) -> AnomalyMessage:
        total_anomalies = []
        if len(tracks) != 0:
            criterion = nn.L1Loss(reduction='sum').to(self._device)

            for id in tracks.keys():
                anomalies=[] 
                if len(tracks[id]) < 5:
                    continue
                with SuppressOutput():      # suppress tqdm progress bar
                    trajectories_dataset = makeTorchPredictionDataSet({id: tracks[id]})

                with torch.no_grad():
                    for trajectory, orig_input in trajectories_dataset:
                        trajectory = trajectory.to(self._device).reshape(-1, trajectory.shape[-2], trajectory.shape[-1])
                        orig_input = orig_input.squeeze(0)
                        target = trajectory
                        pred = self._model(target)
                        loss = criterion(pred, target)
                        if loss.item() > self._config.model.anomaly_loss_threshold:
                            log.info("anomaly found")
                            anomalies.append([trajectory, tracks[id][orig_input[0]: orig_input[1]]])

                total_anomalies += anomalies

        #TODO consider refactorying format of total_anomalies
        return self._write_anomaly_message(total_anomalies, frames)
    
    def _write_anomaly_message(self, total_anomalies, frames) -> AnomalyMessage:
        anomaly_msg = AnomalyMessage()

        if len(total_anomalies) > 0:
            for anomaly in total_anomalies:
                trajectory = anomaly_msg.trajectories.add() 
                #print(anomaly[1])
                #trajectory.CopyFrom(self._map_anomaly(anomaly[1].view(-1, anomaly[1].size(-1))))
                trajectory.CopyFrom(self._map_anomaly(anomaly[1]))
        
            if len(frames) > 0:
                for frame in frames:
                    anomaly_frame = anomaly_msg.jpeg_frames.add()
                    anomaly_frame.frame_data_jpeg = frame.frame_data_jpeg
                    anomaly_frame.timestamp_utc_ms = frame.timestamp_utc_ms
        
        return anomaly_msg

    def _map_anomaly(self, anomaly) -> Trajectory:
        trajectory = Trajectory()

        for track in anomaly:
            #trajectory_point = TrajectoryPoint()
            trajectory_point = trajectory.trajectory_points.add()

            center_point = Point()
            center_point.x = track.get_center()[0] 
            center_point.y = track.get_center()[1]

            trajectory_point.detection_center.CopyFrom(center_point)
            trajectory_point.anomaly_trigger = True
            trajectory_point.timestamp_utc_ms = int(track.capture_ts.timestamp() * 1000)
        
        #TODO: trajectory object_id, class_id

        '''
        trajectory_point = TrajectoryPoint()
        center_point = Point()
        center_point.x = [track.get_center()[0] for track in anomaly]
        center_point.y = [1 - track.get_center()[1] for track in anomaly]
        anomaly_trigger = [True for _ in anomaly]
        timestamp_utc_ms = [track.capture_ts for track in anomaly]

        trajectory = Trajectory()
        trajectory_point = trajectory.trajectory_points.add()
        trajectory_point.detection_center.CopyFrom(center_point)
        trajectory_point.anomaly_trigger.CopyFrom(anomaly_trigger)
        trajectory_point.timestamp_utc_ms.CopyFrom(timestamp_utc_ms)
        '''
        #print(trajectory)
        return trajectory

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
