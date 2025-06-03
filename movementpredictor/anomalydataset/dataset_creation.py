import json
import numpy as np
import random
import zipfile
import pandas as pd
from movementpredictor.data import datamanagement, datafilterer
import os
import re
from collections import defaultdict
from typing import Dict, List
import pybase64
from tqdm import tqdm
from visionapi.sae_pb2 import SaeMessage
from visionlib import saedump


class DetectionData:
    obj_id: str
    class_id: str
    longitude: float
    latitude: float
    boundingbox: List[float]        # [x_min, y_min, x_max, y_max]
    confidence: float
    timestamp: int


def create_train_data(path_sae_dump, path_store):
    detection_dict = []
    os.makedirs(path_store, exist_ok=True)

    trackManager = datamanagement.TrackingDataManager()
    trackedObjects = trackManager.getTrackedBaseData(path_sae_dump, inferencing=False)
    trackedObjects = datafilterer.DataFilterer().apply_filtering(trackedObjects)

    lookup = {
        (obj_id, entry.capture_ts): entry.bbox
        for obj_id, entries in trackedObjects.items()
        for entry in entries
    }
    lookup = defaultdict(lambda: None, lookup)
    
    with open(path_sae_dump, 'r') as input_file:
        messages = saedump.message_splitter(input_file)

        start_message = next(messages)
        saedump.DumpMeta.model_validate_json(start_message)

        for i, message in tqdm(enumerate(messages), desc="collecting frames"):
            event = saedump.Event.model_validate_json(message)
            proto_bytes = pybase64.standard_b64decode(event.data_b64)

            proto = SaeMessage()
            proto.ParseFromString(proto_bytes)

            detections_per_timestamp = []
            for det in proto.detections:
                
                obj_id = str(det.object_id.hex())
                bbox = lookup[(obj_id, proto.frame.timestamp_utc_ms)]
                if bbox is None:
                    continue
                boundingbox = [
                        round(bbox[0][0], 4), round(bbox[0][1], 4),
                        round(bbox[1][0], 4), round(bbox[1][1], 4)
                    ]
                
                detections_per_timestamp.append({
                    "class_id": det.class_id,
                    "object_id": obj_id,
                    "longitude": round(det.geo_coordinate.longitude, 5),
                    "latitude": round(det.geo_coordinate.latitude, 5),
                    "boundingbox": boundingbox,     
                    "confidence": round(det.confidence, 4)
                })

            detection_dict.append({
                "timestamp": proto.frame.timestamp_utc_ms,
                "detections": detections_per_timestamp
            })
    
    path_detection_json = os.path.join(path_store, "object_detections.json")
    with open(path_detection_json, "w", encoding="utf-8") as file:
        json.dump(detection_dict, file, indent=4)

    


def store_data_sniplet(frames_with_ts, detection_data: List[DetectionData], base_path: str, ids_of_interest: List[str], labels: List[int], 
                       time_intervals: List[List[int]], prefix="video", width=3):
    #TODO: talk about data format with Flo

    # create subfolder videoxxx
    os.makedirs(base_path, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")

    max_idx = -1
    for name in os.listdir(base_path):
        match = pattern.match(name)
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx

    next_idx = max_idx + 1
    new_name = f"{prefix}{next_idx:0{width}d}"
    new_folder_path = os.path.join(base_path, new_name)
    frames_path = os.path.join(new_folder_path, "video_frames")

    os.makedirs(frames_path, exist_ok=True)

    # store frames and detection data
    data_dict = {
        "name_video": new_name,
        "frame_data": []
    }

    detection_data_dict = defaultdict(list)
    for det in detection_data:
        key = det.timestamp
        detection_data_dict[key].append(det)

    #with zipfile.ZipFile(frames_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
    for idx, (frame_bytes, ts) in enumerate(frames_with_ts, start=1):
        filename = f"frame{idx:04d}.jpg"
        file_path = os.path.join(frames_path, filename)
        #zipf.writestr(filename, frame_bytes)
        with open(file_path, "wb") as f:
            f.write(frame_bytes)

        detections = detection_data_dict[ts]
        detection_dicts = []

        for det in detections:
            detection_dicts.append({
                "class_id": det.class_id,
                "object_id": det.obj_id,
                "longitude": round(det.longitude, 4),
                "latitude": round(det.latitude, 4),
                "boundingbox": det.boundingbox,     
                "confidence": round(det.confidence, 4)
                })

        frame_data_entry = {
            "frame": filename,
            "timestamp": ts,
            "detections": detection_dicts
        }
        data_dict["frame_data"].append(frame_data_entry)
    
    path_detection_json = os.path.join(new_folder_path, "object_detections.json")
    with open(path_detection_json, "w", encoding="utf-8") as file:
        json.dump(data_dict, file, indent=4)
    
    # store anomaly_predictions
    anomaly_intervals = []
    object_ids_in_this_video = np.unique(np.array([det.obj_id for det in detection_data]))
    for obj_id, label, interval in zip(ids_of_interest, labels, time_intervals):
        #if obj_id in detection_data_dict.keys():
         #   print(obj_id, "         ", label, "         ", interval)
        if obj_id in object_ids_in_this_video and label != 0:
            anomaly_intervals.append({
                "object_id": obj_id,
                "start_timestamp": interval[0],
                "end_timestamp": interval[1],
                "label": label
            })

    df = pd.DataFrame(anomaly_intervals)
    df.to_csv(os.path.join(new_folder_path, "labels.csv"), index=False)



def create_datasniplets(ids_of_interest: List[str], labels: List[int], anomaly_intervals: List[List[int]], path_sae_dump: str, path_store: str,
                        length_vids_in_min=3):
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

        if delta_ms >= length_vids_in_min * 60 * 1000:  # length_vids_in_min in millisec
            end = necessary_timestamps[i-1]
            time_intervals.append([start, end])
            start = necessary_timestamps[i] 
    
    print(len(time_intervals))
    time_intervals.append([start, necessary_timestamps[-1]])

    for interval in time_intervals:
        start, end = interval
        before = random.uniform(0, (length_vids_in_min/2) * 60 * 1000)      # add random max. length_vids_in_min/2 min before & max length_vids_in_min/2 min after interval
        after = random.uniform(0, (length_vids_in_min/2) * 60 * 1000)
        interval = [start-before, end+after]
    
    frames_with_ts = []
    detection_data: List[DetectionData] = []
    num_interval = 0
    os.makedirs(path_store, exist_ok=True)
    
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
                frames_with_ts.append([proto.frame.frame_data_jpeg, ts])

                for detection in proto.detections:
                    det = DetectionData()

                    det.class_id = detection.class_id
                    det.obj_id = str(detection.object_id.hex())
                    det.longitude = detection.geo_coordinate.longitude
                    det.latitude = detection.geo_coordinate.latitude
                    det.confidence = detection.confidence
                    det.timestamp = ts

                    bbox = detection.bounding_box
                    det.boundingbox = [bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y]

                    detection_data.append(det)
            
            else:       # ts > interval end time_intervals[num_interval][1]
        
                # BOUNDINGBOX SMOOTHING
                track_list = []
                for det in detection_data:
                    center = ((det.boundingbox[0] + det.boundingbox[2]) * 0.5, (det.boundingbox[1] + det.boundingbox[3]) * 0.5)
                    bbox = [[det.boundingbox[0], det.boundingbox[1]], [det.boundingbox[2], det.boundingbox[3]]]
                    tracked_object = datamanagement.TrackedObjectPosition(capture_ts=det.timestamp, uuid=det.obj_id, class_id=det.class_id, center=center, bbox=bbox)
                    track_list.append(tracked_object)
                
                trackedObjects = datafilterer.DataFilterer().only_smoothing(track_list)
                lookup = {
                    (det.obj_id, det.timestamp): idx
                    for idx, det in enumerate(detection_data)
                }
                for obj_id, track_list in tqdm(trackedObjects.items(), desc="replace bboxes with smoothed bbox"):
                    for trk in track_list:
                        key = (obj_id, trk.capture_ts)
                        idx = lookup.get(key)
                        if idx is not None:
                            detection_data[idx].boundingbox = [
                                round(trk.bbox[0][0], 4), round(trk.bbox[0][1], 4),
                                round(trk.bbox[1][0], 4), round(trk.bbox[1][1], 4)
                            ]
                        else:
                            print("ERROR")
                            exit(1)

                # store dataset
                store_data_sniplet(frames_with_ts, detection_data, path_store, ids_of_interest, labels, anomaly_intervals)
                num_interval += 1
                frames_with_ts = []
                detection_data = []


def get_detected_anomalies_from_label_box(path_label_box: str, camera: str) -> List[str]:
    path_label_box = os.path.join(path_label_box, camera)
    ids = os.listdir(path_label_box)
    labels = []
    time_intervals = []
    for obj_id in ids:
        with open(os.path.join(path_label_box, obj_id, "labeldata.json"), "r", encoding="utf-8") as f:
            label_data = json.load(f) 
        labels.append(label_data["label"])
        time_intervals.append(label_data["time_interval"])
    return ids, labels, time_intervals

