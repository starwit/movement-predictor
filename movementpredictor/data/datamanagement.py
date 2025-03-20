
from movementpredictor.data.trackedobjectposition import TrackedObjectPosition
from tqdm import tqdm
import logging
import pybase64
import numpy as np
import cv2
import torch

from visionapi.sae_pb2 import SaeMessage
from visionlib import saedump

log = logging.getLogger(__name__)


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


def getTrackedBaseData(path, num_batch) -> list[TrackedObjectPosition]:
    try:
        extracted_tracks = [] 
        with open(path, 'r') as input_file:
            messages = saedump.message_splitter(input_file)

            start_message = next(messages)
            dump_meta = saedump.DumpMeta.model_validate_json(start_message)
            print(f'Starting playback from file {path} containing streams {dump_meta.recorded_streams}')

            '''don't pick the data in their order because one dataset might only contain images at night -> inequal distributions'''
            for count, message in tqdm(enumerate(messages)):

                #if count > 10000:
                 #   break

                if  count >= num_batch*100000 and (count < (num_batch+1)*100000 or num_batch == 11):
                    event = saedump.Event.model_validate_json(message)
                    proto_bytes = pybase64.standard_b64decode(event.data_b64)

                    proto = SaeMessage()
                    proto.ParseFromString(proto_bytes)

                    detections = proto.detections

                    for detection in detections:

                        bbox = detection.bounding_box
                        classes_of_interest = [2, 5, 7]

                        # only cars, trucks and busses for now and bbox not too small
                        if detection.class_id in classes_of_interest and bbox.max_x - bbox.min_x > 0.02 and bbox.max_y - bbox.min_y > 0.02: 

                            track = TrackedObjectPosition()
                            track.set_capture_ts(proto.frame.timestamp_utc_ms)
                            track.set_uuid(detection.object_id.hex())
                            track.set_class_id(detection.class_id)
                            track.set_center((bbox.min_x + (bbox.max_x-bbox.min_x)/2, bbox.min_y + (bbox.max_y-bbox.min_y)/2))
                            track.set_bbox([[bbox.min_x, bbox.min_y], [bbox.max_x, bbox.max_y]])
                            extracted_tracks.append(track)

                if count >= (num_batch+1)*100000:
                    break

        return extracted_tracks

    except Exception as e:

        print(f"Error processing the file: {e}")
        return None
