import random
from movementpredictor.evaluation.eval_config import EvalConfig
from movementpredictor.anomalydetection.anomaly_detector import find_intervals_containing_timestamp
from movementpredictor.anomalydetection.video_generation import store_video
from movementpredictor.evaluation.eval_data import get_obj_ids_from_label_box
import os
from collections import defaultdict
from typing import Dict, List
import pybase64
from tqdm import tqdm
from visionapi.sae_pb2 import SaeMessage
from visionlib import saedump

evalconfig = EvalConfig()

class DetectionData:
    obj_id: str
    class_id: str
    #geo_coord: str
    boundingbox: List[float]        # [x_min, y_min, x_max, y_max]
    confidence: float
    timestamp: int


def store_data_sniplet(frames: List[bytes], detection_data: List[DetectionData], path):
    #TODO: talk about data format with Flo
    ...


def create_datasniplets(ids_of_interest: List[str], path_sae_dump: str, path_store: str, camera:str, length_vids_in_min=5):
    necessary_timestamps = []

    with open(path_sae_dump, 'r') as input_file:
        messages = saedump.message_splitter(input_file)

        start_message = next(messages)
        saedump.DumpMeta.model_validate_json(start_message)

        for message in tqdm(messages, desc="collecting frames"):
            event = saedump.Event.model_validate_json(message)
            proto_bytes = pybase64.standard_b64decode(event.data_b64)

            proto = SaeMessage()
            proto.ParseFromString(proto_bytes)

            for detection in proto.detections:
                id = str(detection.object_id.hex())
                if id in ids_of_interest:
                    necessary_timestamps.append(proto.frame.timestamp_utc_ms)
                    break
    
    necessary_timestamps = sorted(necessary_timestamps)
    time_intervals = []

    start = necessary_timestamps[0]

    for i in range(1, len(necessary_timestamps)):
        delta_ms = necessary_timestamps[i] - necessary_timestamps[i-1]

        if delta_ms >= length_vids_in_min * 60 * 1000:  # 5 min in millisec
            end = necessary_timestamps[i-1]
            time_intervals.append([start, end])
            start = necessary_timestamps[i] 
    
    print(len(time_intervals))
    exit(0)

    time_intervals.append([start, necessary_timestamps[-1]])

    for interval in time_intervals:
        start, end = interval
        before = random.uniform(0, (length_vids_in_min/2) * 60 * 1000)      # add random max. length_vids_in_min/2 min before & max length_vids_in_min/2 min after interval
        after = random.uniform(0, (length_vids_in_min/2) * 60 * 1000)
        interval = [start-before, end+after]
    
    frames = []
    detection_data: List[DetectionData] = []
    num_interval = 0
    path = os.path.join(path_store, camera)
    os.makedirs(path, exist_ok=True)
    
    with open(path_sae_dump, 'r') as input_file:
        messages = saedump.message_splitter(input_file)

        start_message = next(messages)
        saedump.DumpMeta.model_validate_json(start_message)

        for message in tqdm(messages, desc="collecting frames"):
            event = saedump.Event.model_validate_json(message)
            proto_bytes = pybase64.standard_b64decode(event.data_b64)

            proto = SaeMessage()
            proto.ParseFromString(proto_bytes)

            ts = proto.frame.timestamp_utc_ms 
            if ts < time_intervals[num_interval][0]:
                continue

            elif ts >= time_intervals[num_interval][0] and ts <= time_intervals[num_interval][1]:
                frames.append(proto.frame.frame_data_jpeg)

                for detection in proto.detections:
                    det = DetectionData()

                    det.class_id = detection.class_id
                    det.obj_id = str(detection.object_id.hex())
                    #det.geo_coord = detection.geo_coordinate
                    det.confidence = detection.confidence
                    det.timestamp = ts

                    bbox = detection.bounding_box
                    det.boundingbox = [bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y]

                    detection_data.append(det)
            
            else:       # ts > interval end time_intervals[num_interval][1]
                store_data_sniplet(frames, detection_data, os.path.join(path, str(time_intervals[num_interval][0])))
                num_interval += 1
                frames = []
                detection_data = []


def main():
    ids_of_interest = get_obj_ids_from_label_box(evalconfig.path_label_box, evalconfig.camera)
    create_datasniplets(ids_of_interest, evalconfig.path_sae_dump, "movementpredictor/evaluation/anomaly_dataset", evalconfig.camera)


if __name__ == "__main__":
    main()