import logging
import os
import sys
import ast
from typing import List
import torch

from movementpredictor import DataFilterer, makeTorchDataLoader, CNN, inference_with_stats, get_meaningful_unlikely_samples, InferenceResult
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
        self.model = CNN().to(self._device)

        try:
            weights = torch.load(self._config.model.weights_path, map_location=torch.device(self._device))
        except:
            log.error(f"Model weights not found at {self._config.model.weights_path}")
            exit(1)
        try:    
            self.model.load_state_dict(weights)
        except:
            log.error(f"Model weights do not fit model architecture")
            exit(1)

        log.info("starting anomaly detection with parameters:")
        for key in self._parameters.keys():
            log.info(key + ": " + str(self._parameters[key]))
      
    def filter_tracks(self, tracks):
        if len(tracks) != 0:
            log.info(f"num tracks before filtering: {len(tracks)}")
            if self._config.filtering:
                tracks = DataFilterer().apply_filtering(tracks)
                log.info(f"num tracks after filtering: {len(tracks)}")
            else:
                tracks = DataFilterer().only_smoothing(tracks)
                log.info(f"TESTING WITHOUT FILTERING")
        return tracks
    
    def examine_tracks_for_anomalies(self, tracks, frames) -> AnomalyMessage:
        if len(tracks) != 0:
            with SuppressOutput():      # suppress tqdm progress bar
                dataloader = makeTorchDataLoader(tracks, self._config.model.background_path)
                outputs_with_stats = inference_with_stats(self.model, dataloader)
                anomalies = get_meaningful_unlikely_samples(outputs_with_stats, self._parameters["anomaly_threshold"] 
                                                            if self._config.model.anomaly_threshold_override is None else self._config.model.anomaly_threshold_override)

                if len(anomalies) != 0:
                    log.info("anomaly found")

        return self._write_anomaly_message(anomalies, frames, tracks)
    
    def _write_anomaly_message(self, anomalies: List[InferenceResult], frames, tracks) -> AnomalyMessage:
        anomaly_msg = AnomalyMessage()
        
        if len(anomalies) > 0:
            anomaly_trajectories = {}
            anomaly_ids = [anomaly.obj_id for anomaly in anomalies]

            for id in anomaly_ids:
                anomaly_trajectories[id] = tracks[ast.literal_eval(id)]

            for id in anomaly_trajectories.keys():
                trajectory = anomaly_msg.trajectories.add() 
                trajectory.CopyFrom(self._map_anomaly(anomalies, anomaly_trajectories[id]))
        
            if len(frames) > 0:
                for frame in frames:
                    anomaly_frame = anomaly_msg.jpeg_frames.add()
                    anomaly_frame.frame_data_jpeg = frame.frame_data_jpeg
                    anomaly_frame.timestamp_utc_ms = frame.timestamp_utc_ms
        
        return anomaly_msg

    def _map_anomaly(self, total_anomalies: List[InferenceResult], anomaly_trajectory) -> Trajectory:
        trajectory = Trajectory()
        trajectory.object_id = anomaly_trajectory[0].get_uuid()
        trajectory.class_id = anomaly_trajectory[0].get_class_id()

        anomaly_trigger_ts = [a.timestamp for a in total_anomalies if a.obj_id == trajectory.object_id]

        for track in anomaly_trajectory:
            trajectory_point = trajectory.trajectory_points.add()

            center_point = Point()
            center_point.x = track.get_center()[0] 
            center_point.y = track.get_center()[1]

            trajectory_point.detection_center.CopyFrom(center_point)
            if track.get_capture_ts() in anomaly_trigger_ts:
                trajectory_point.anomaly_trigger = True
            else:
                trajectory_point.anomaly_trigger = False
            trajectory_point.timestamp_utc_ms = track.get_capture_ts()

        return trajectory
