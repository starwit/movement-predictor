import os
from tqdm import tqdm
import logging
import pybase64
import numpy as np
import cv2
import torch
from typing import List, Optional
from pydantic import BaseModel

from visionapi.sae_pb2 import SaeMessage
from visionlib import saedump

log = logging.getLogger(__name__)


class TrackedObjectPosition(BaseModel):
    capture_ts: int 
    uuid: str 
    class_id: int 
    center: List[float]  # [x, y]
    bbox: List[List[float]]  # [[x1, y1], [x2, y2]]
    clear_detection: bool = False
    movement_angle: Optional[float] = None 
   # movement_speed: Optional[float] = None 


def get_downsampled_tensor_img(frame, pixel):

        frame_data = frame.frame_data_jpeg
        np_arr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, (pixel, pixel), interpolation=cv2.INTER_AREA)
        tensor_img = torch.from_numpy(resized_img).float() / 255.0

        return tensor_img


def get_background_frame(path, pixel):
        try:
            tensor_img = None

            with open(path, 'r') as input_file:
                messages = saedump.message_splitter(input_file)

                start_message = next(messages)
                saedump.DumpMeta.model_validate_json(start_message)

                for count, message in enumerate(messages):
                    event = saedump.Event.model_validate_json(message)
                    proto_bytes = pybase64.standard_b64decode(event.data_b64)

                    proto = SaeMessage()
                    proto.ParseFromString(proto_bytes)

                    tensor_img = get_downsampled_tensor_img(proto.frame, pixel)
                    return tensor_img

        
        except Exception as e:

            print(f"Error processing the file: {e}")
            return None
        

def store_frame(frame: torch.Tensor, path_store):
    torch.save(frame, os.path.join(path_store, "frame.pth"))


def load_background_frame(path_store):
    return torch.load(os.path.join(path_store, "frame.pth"))
        

class TrackingDataManager:
    border_threshold = None
    frame_rate = None
    start_timestamp = None

    def getTrackedBaseData(self, path, inferencing=True) -> list[TrackedObjectPosition]:
        try:
            extracted_tracks = [] 
            with open(path, 'r') as input_file:

                messages = saedump.message_splitter(input_file)

                start_message = next(messages)
                dump_meta = saedump.DumpMeta.model_validate_json(start_message)
                log.info(f'Starting playback from file {path} containing streams {dump_meta.recorded_streams}')
                frames_count = 0
                
                for count, message in tqdm(enumerate(messages)):
                    #if count > 10000:
                     #   break
                    event = saedump.Event.model_validate_json(message)
                    proto_bytes = pybase64.standard_b64decode(event.data_b64)

                    proto = SaeMessage()
                    proto.ParseFromString(proto_bytes)

                    if self.frame_rate is None:
                        if self.start_timestamp is None:
                            self.start_timestamp = proto.frame.timestamp_utc_ms
                        else:
                            frames_count += 1
                            if frames_count == 10:
                                diff = proto.frame.timestamp_utc_ms - self.start_timestamp
                                self.frame_rate = 1000*10/diff
                                print("frame-rate=", self.frame_rate)
                    
                    extracted_tracks = self.extract_tracked_objects(proto, extracted_tracks, inferencing)

            return extracted_tracks

        except Exception as e:

            print(f"Error processing the file: {e}")
            return None


    def extract_tracked_objects(self, proto, extracted_tracks=[], inferencing=True):
        if self.border_threshold is None:
            self._calc_frame_border_threshold(proto.frame)

        outer_threshold_x = 1 - self.border_threshold[0]
        outer_threshold_y = 1 - self.border_threshold[1]

        for det in proto.detections:
            min_x = det.bounding_box.min_x
            min_y = det.bounding_box.min_y
            max_x = det.bounding_box.max_x
            max_y = det.bounding_box.max_y

            classes_of_interest = [0, 1, 2, 3, 5, 7]        # person, bicycle, car, motorcycle, bus, truck
            min_bbox_size = 0.03 if inferencing else 0.02

            # only cars, trucks and busses for now and bbox not too small
            if det.class_id in classes_of_interest and ((max_x - min_x > min_bbox_size and max_y - min_y > min_bbox_size) or det.class_id != 2): 

                if inferencing and (min_x <= self.border_threshold[0] or min_y <= self.border_threshold[1] or max_x >= outer_threshold_x or max_y >= outer_threshold_y):
                    continue

                capture_ts = proto.frame.timestamp_utc_ms
                uuid = det.object_id.hex()
                class_id = det.class_id
                center = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5)
                bbox = [[min_x, min_y], [max_x, max_y]]
                track = TrackedObjectPosition(capture_ts=capture_ts, uuid=uuid, class_id=class_id, center=center, bbox=bbox)

                extracted_tracks.append(track)
        
        return extracted_tracks

    def _calc_frame_border_threshold(self, frame):
        width_height = frame.shape.width / frame.shape.height
        self.border_threshold = [0.05, width_height * 0.05]
