
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
    movement_angle: Optional[float] = None 


def get_downsampled_tensor_img(frame, dim_x, dim_y):

        frame_data = frame.frame_data_jpeg
        np_arr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, (dim_x, dim_y), interpolation=cv2.INTER_AREA)
        tensor_img = torch.from_numpy(resized_img).float() / 255.0

        return tensor_img


def get_background_frame(path, dim_x, dim_y):
        # TODO: choose frame with least amount of detections and good quality
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

                #    if len(proto.detections) < 10:
                    tensor_img = get_downsampled_tensor_img(proto.frame, dim_x, dim_y)
                    return tensor_img

                #return get_downsampled_tensor_img(proto.frame, dim_x, dim_y)
        
        except Exception as e:

            print(f"Error processing the file: {e}")
            return None
        

class TrackingDataManager:
    border_threshold = None
    frame_rate = None

    def getTrackedBaseData(self, path, num_batch=None, inferencing=True) -> list[TrackedObjectPosition]:
        try:
            extracted_tracks = [] 
            with open(path, 'r') as input_file:
                messages = saedump.message_splitter(input_file)

                start_message = next(messages)
                dump_meta = saedump.DumpMeta.model_validate_json(start_message)
                print(f'Starting playback from file {path} containing streams {dump_meta.recorded_streams}')

                start_timestamp = None

                '''don't pick the data in their order because one dataset might only contain images at night -> inequal distributions'''
                for count, message in tqdm(enumerate(messages)):

                    #if count > 1000:
                     #   break

                    # TODO: 1000000 entfernen und die Werte irgendiwe anders in 3 Datensätze aufteilen
                    if  num_batch is None or (count >= num_batch*100000 and (count < (num_batch+1)*100000 or num_batch == 11)):
                        event = saedump.Event.model_validate_json(message)
                        proto_bytes = pybase64.standard_b64decode(event.data_b64)

                        proto = SaeMessage()
                        proto.ParseFromString(proto_bytes)

                        if self.frame_rate is None:
                            if start_timestamp is None:
                                start_timestamp = proto.frame.timestamp_utc_ms
                            else:
                                diff = proto.frame.timestamp_utc_ms - start_timestamp
                                self.frame_rate = 1000/diff

                        extracted_tracks = self.extract_tracked_objects(proto, extracted_tracks, inferencing)

                    if num_batch is not None and count >= (num_batch+1)*100000:
                        break

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
            if det.class_id in classes_of_interest and max_x - min_x > min_bbox_size and max_y - min_y > min_bbox_size: 

                if inferencing and (min_x <= self.border_threshold[0] or min_y <= self.border_threshold[1] or max_x >= outer_threshold_x or max_y >= outer_threshold_y):
                    continue

                #track = TrackedObjectPosition()
                #track.set_capture_ts(proto.frame.timestamp_utc_ms)
                #track.set_uuid(det.object_id.hex())
                #track.set_class_id(det.class_id)
                #track.set_center(((min_x + max_x) * 0.5, (min_y + max_y) * 0.5))
                #track.set_bbox([[min_x, min_y], [max_x, max_y]])
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
