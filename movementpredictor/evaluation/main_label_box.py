import sys
sys.path.append('/home/starwit01/workspaces/hanna/movement-predictor')

from movementpredictor.evaluation.eval_config import EvalConfig
from movementpredictor.anomalydetection.anomaly_detector import find_intervals_containing_timestamp
from movementpredictor.anomalydetection.video_generation import store_video

import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import pybase64
from tqdm import tqdm
import json
from visionapi.sae_pb2 import SaeMessage
from visionlib import saedump
import logging


log = logging.getLogger(__name__)
evalconfig = EvalConfig()



def extract_predictions(path_stored_predictions: str):
    predictions_folder = Path(path_stored_predictions)
    predictions = []

    for json_file in predictions_folder.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                anomaly_predictions = json.load(f)
                predictions.append(anomaly_predictions["predictions"])
                
            except json.JSONDecodeError as e:
                log.error(f"could not load json file {json_file.name}: {e}")
                exit(1)

    return predictions


def store_for_labelling(predicted_anomalies: Dict[str, Dict[str, List]], camera: str, path_label_storing: str, path_sae_dumps: List[str]):
    not_stored_already = defaultdict()
    path_label_storing = os.path.join(path_label_storing, camera)

    for obj_id in predicted_anomalies.keys():

        timestamps = [int(ts) for ts in predicted_anomalies[obj_id]["timestamps"]]
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        path = os.path.join(path_label_storing, obj_id)

        if os.path.exists(path):
            json_path = os.path.join(path, "labeldata.json")

            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    label_data = json.load(f) 

                if min_ts >= label_data["time_interval"][0] and max_ts <= label_data["time_interval"][1]:
                    continue

            else: 
                log.error("Missing file " + json_path + "! folder for this predicted anomaly will be generated again.")
            
        else:
            os.makedirs(path, exist_ok=True)
            label_data = defaultdict()
            label_data["obj_id"] = obj_id
            label_data["label"] = "None"
        
        label_data["time_interval"] = [min_ts, max_ts]
        with open(os.path.join(path, "labeldata.json"), "w", encoding="utf-8") as file:
            json.dump(label_data, file, indent=4)

        not_stored_already[obj_id] = predicted_anomalies[obj_id]
    
    if len(not_stored_already) > 0:
        create_video(not_stored_already, path_sae_dumps, path_label_storing)


def create_video(anomaly_dict: Dict[str, Dict[str, List]], path_sae_dumps: List[str], path_store: str):
    video_dict = defaultdict(list)

    for key in anomaly_dict.keys():
        timestamps = [int(ts) for ts in anomaly_dict[key]["timestamps"]]
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        start = min_ts - 5000
        end = max_ts + 5000
        video_dict[(key, start, end)] = []
    
    for dump_path in path_sae_dumps:
        with open(dump_path, 'r') as input_file:
            messages = saedump.message_splitter(input_file)

            start_message = next(messages)
            saedump.DumpMeta.model_validate_json(start_message)

            for message in tqdm(messages, desc="collecting frames"):
                event = saedump.Event.model_validate_json(message)
                proto_bytes = pybase64.standard_b64decode(event.data_b64)

                proto = SaeMessage()
                proto.ParseFromString(proto_bytes)
                frame_ts = proto.frame.timestamp_utc_ms

                fitting_keys = find_intervals_containing_timestamp(frame_ts, video_dict.keys())

                for key in fitting_keys:
                    frame_info = [proto.frame, None]
                    for detection in proto.detections:
                        if str(key[0]) == str(detection.object_id.hex()):
                            frame_info[1] = detection.bounding_box
                            break
                    video_dict[key].append(frame_info)
    
    for key in video_dict:
        path = os.path.join(path_store, key[0])
        store_video(video_dict[key], path)



def main():
    # get predictions from stored prediction data
    predictions = extract_predictions(evalconfig.path_store_anomalies)

    # store videos for labelling
    for i, predictions_per_method in enumerate(predictions):
        log.info(f"start video creation {i} of {len(predictions)}")
        store_for_labelling(predictions_per_method, evalconfig.camera, evalconfig.path_label_box, evalconfig.path_sae_dumps)


if __name__ == "__main__":
    main()