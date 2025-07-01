from movementpredictor.evaluation.eval_config import EvalConfig
from movementpredictor.anomalydetection.anomaly_detector import find_intervals_containing_timestamp
from movementpredictor.anomalydetection.video_generation import store_video

import os
import cv2
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import pybase64
from tqdm import tqdm
import json
import numpy as np
from visionapi.sae_pb2 import SaeMessage
from visionlib import saedump
import logging
from PIL import Image


log = logging.getLogger(__name__)
evalconfig = EvalConfig()


def extract_predictions(path_stored_predictions: str):
    predictions_folder = Path(path_stored_predictions)
    predictions = []

    for json_file in predictions_folder.glob("*.json"):
        if json_file.name.endswith("100.json"):
            continue

        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                anomaly_predictions = json.load(f)
                predictions.append(anomaly_predictions["predictions"])
                
            except json.JSONDecodeError as e:
                log.error(f"could not load json file {json_file.name}: {e}")
                exit(1)

    return predictions


def store_for_labelling(predicted_anomalies: Dict[str, Dict[str, List]], camera: str, path_label_storing: str):
    not_stored_already = defaultdict()
    #name_sae_dump = os.path.basename(path_sae_dump)
    path_label_storing = os.path.join(path_label_storing, camera)#, name_sae_dump)

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

                if len(label_data["time_interval"]) > 0 and min_ts >= label_data["time_interval"][0] and max_ts <= label_data["time_interval"][1]:
                    continue

                else: 
                    min_ts = min(label_data["time_interval"][0], min_ts) if len(label_data["time_interval"]) > 0 else min_ts
                    max_ts = max(label_data["time_interval"][1], max_ts) if len(label_data["time_interval"]) > 0 else max_ts

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

        #not_stored_already[obj_id] = predicted_anomalies[obj_id]
        not_stored_already[obj_id] = [int(min_ts), int(max_ts)]

    # return intervals; video creation deferred to main
    return not_stored_already


def create_video(anomaly_dict: Dict[str, List[int]], path_sae_dumps: str, path_store: str):
    video_dict = defaultdict(list)

    for key in anomaly_dict.keys():
        #timestamps = [int(ts) for ts in anomaly_dict[key]["timestamps"]]
        min_ts = anomaly_dict[key][0]
        max_ts = anomaly_dict[key][1]
        start = min_ts - 5000
        end = max_ts + 5000
        video_dict[(key, start, end)] = []
    
    for path_sae_dump in path_sae_dumps:
        with open(path_sae_dump, 'r') as input_file:
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
        store_frames_and_bboxs(video_dict[key], path)
        store_video(video_dict[key], path)


def store_frames_and_bboxs(frame_infos, path):
    frames = [frame_info[0] for frame_info in frame_infos]
    bboxs = [frame_info[1] for frame_info in frame_infos]

    filtered_frames = []
    filtered_boxes  = []

    for bbox, frame in zip(bboxs, frames):
        if bbox is not None:
            filtered_frames.append(frame)
            filtered_boxes.append(bbox)

    rgb_frames = [get_downsampled_pil_img(frame) for frame in filtered_frames]

    frames_path = os.path.join(path, "frames")
    os.makedirs(frames_path, exist_ok=True)

    anno = {}
    for idx, (pil_img, bbox) in enumerate(zip(rgb_frames, filtered_boxes), start=1):
        fname = f"{idx:05d}.jpg"
        frame_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(frames_path, fname), frame_np)  
        anno[fname] = [bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y]

    with open(os.path.join(path, "bboxes.json"), "w") as f:
        json.dump(anno, f, indent=2)


def get_downsampled_pil_img(frame, pixel=224):
    np_arr = np.frombuffer(frame.frame_data_jpeg, dtype=np.uint8)

    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(img_rgb, (pixel, pixel), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized)


def main():

    predictions = extract_predictions(evalconfig.path_store_anomalies)

    all_intervals = {}
    for i, preds in enumerate(predictions):

        log.info(f"processing predictions {i+1}/{len(predictions)}")
        new_intervals = store_for_labelling(
            preds,
            evalconfig.camera,
            evalconfig.path_label_box
        )

        for obj_id, (min_ts, max_ts) in new_intervals.items():
            if obj_id in all_intervals:
                all_intervals[obj_id][0] = min(all_intervals[obj_id][0], min_ts)
                all_intervals[obj_id][1] = max(all_intervals[obj_id][1], max_ts)
            else:
                all_intervals[obj_id] = [min_ts, max_ts]

    if all_intervals:
        path_label_store = os.path.join(evalconfig.path_label_box, evalconfig.camera)
        create_video(all_intervals, evalconfig.path_sae_dumps, path_label_store)

if __name__ == "__main__":
    main()