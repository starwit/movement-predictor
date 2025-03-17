import logging
from typing import Any

from prometheus_client import Counter, Histogram, Summary
from visionapi.sae_pb2 import SaeMessage

from anomalydetection.detector import Detector
from anomalydetection.modelinfocollector import ModelInfoCollector
from anomalydetection.trackcollector import TrackCollector

from .config import AnomalyDetectionConfig

logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s')
logger = logging.getLogger(__name__)

GET_DURATION = Histogram('anomaly_detection_get_duration', 'The time it takes to deserialize the proto until returning the tranformed result as a serialized proto',
                         buckets=(0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25))
OBJECT_COUNTER = Counter('anomaly_detection_object_counter', 'How many detections have been transformed')
PROTO_SERIALIZATION_DURATION = Summary('anomaly_detection_proto_serialization_duration', 'The time it takes to create a serialized output proto')
PROTO_DESERIALIZATION_DURATION = Summary('anomaly_detection_proto_deserialization_duration', 'The time it takes to deserialize an input proto')


class AnomalyDetection:
    def __init__(self, CONFIG: AnomalyDetectionConfig) -> None:
        logger.setLevel(CONFIG.log_level.value)
        self._config = CONFIG
        self.model_info = ModelInfoCollector(CONFIG).model_info
        self._detector = Detector(self._config)
        self._timed_data_collector = TrackCollector(self._config.log_level.value)
 
    def __call__(self, input_proto) -> Any:
        return self.get(input_proto)
   
    @GET_DURATION.time()
    def get(self, input_proto):
        sae_msg = self._unpack_proto(input_proto)
        #Get anomalies
        self._timed_data_collector.add(sae_msg)
        tracks, frames = self._timed_data_collector.get_latest_data()
        if len(tracks) == 0:
            return None
        filtered_data = self._detector.filter_tracks(tracks)
        if len(filtered_data) == 0:
            return None
        anomaly_message = self._detector.examine_tracks_for_anomalies(filtered_data, frames)

        if len(anomaly_message.trajectories) == 0:
            return None
        
        # Forward camera geo location (unset value does not need special treatment)
        anomaly_message.camera_location.CopyFrom(sae_msg.frame.camera_location)

        return self._create_output(anomaly_message)
    
    @PROTO_DESERIALIZATION_DURATION.time()
    def _unpack_proto(self, sae_message_bytes):
        sae_msg = SaeMessage()
        sae_msg.ParseFromString(sae_message_bytes)
        return sae_msg
    
    @PROTO_SERIALIZATION_DURATION.time()
    def _create_output(self, output_anomaly_msg):

        output_anomaly_msg.model_info.CopyFrom(self.model_info)
        return output_anomaly_msg.SerializeToString()
    