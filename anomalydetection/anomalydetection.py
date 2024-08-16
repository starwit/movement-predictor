import logging
import time
from typing import Any, Dict, NamedTuple

from prometheus_client import Counter, Histogram, Summary
from visionapi.messages_pb2 import BoundingBox, SaeMessage
from visionapi.anomaly_pb2 import AnomalyMessage
from anomalydetection.detector import Detector
from anomalydetection.trajectorycollector import TimedTrajectories

from .config import AnomalyDetectionConfig

logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s')
logger = logging.getLogger(__name__)

GET_DURATION = Histogram('anomaly_detection_get_duration', 'The time it takes to deserialize the proto until returning the tranformed result as a serialized proto',
                         buckets=(0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25))
OBJECT_COUNTER = Counter('anomaly_detection_object_counter', 'How many detections have been transformed')
PROTO_SERIALIZATION_DURATION = Summary('anomaly_detection_proto_serialization_duration', 'The time it takes to create a serialized output proto')
PROTO_DESERIALIZATION_DURATION = Summary('anomaly_detection_proto_deserialization_duration', 'The time it takes to deserialize an input proto')


class AnomalyDetection:
    def __init__(self, config: AnomalyDetectionConfig) -> None:
        self.config = config
        logger.setLevel(self.config.log_level.value)
 
    def __call__(self, input_proto) -> Any:
        return self.get(input_proto)
    
    @GET_DURATION.time()
    def get(self, input_proto):
        sae_msg = self._unpack_proto(input_proto)
        inference_start = time.monotonic_ns()

        # Your implementation goes (mostly) here
        #logger.warning('Received SAE message from pipeline')

        #Get anomalies
        self.timed_data_collector.add(input_proto)
        data = self.timed_data_collector.get_latest_Trajectories()
        frames = self.timed_data_collector.frames
        filtered_data = self.detector.filter_tracks(data)
        total_anomalies = self._get_anomalies(filtered_data, frames)
        
        #TODO move it to anomaly post processing
        self.detector.write_anomalies_to_filesystem(total_anomalies, filtered_data, frames)

        #return self._pack_proto(sae_msg)
        inference_time_us = (time.monotonic_ns() - inference_start) // 1000
        return self._create_output(total_anomalies, sae_msg, inference_time_us)
    
    def _get_anomalies(self, filtered_data, frames):
        total_anomalies = []
        if len(filtered_data) != 0:
            total_anomalies = self.detector.examine(filtered_data, frames)
        return total_anomalies
    
    def _setup(self):
        logger.info(f'Setup Anomaly Detection')
        conf = self.config
        self.detector = Detector(conf)
        self.timed_data_collector = TimedTrajectories(conf.log_level.value, timeout=3)

        
    @PROTO_DESERIALIZATION_DURATION.time()
    def _unpack_proto(self, sae_message_bytes):
        sae_msg = SaeMessage()
        sae_msg.ParseFromString(sae_message_bytes)
        return sae_msg
    
    @PROTO_SERIALIZATION_DURATION.time()
    def _create_output(self, total_anomalies, input_sae_msg: SaeMessage, inference_time_us):
        output_anomaly_msg = AnomalyMessage()
        
        return output_anomaly_msg.SerializeToString()
