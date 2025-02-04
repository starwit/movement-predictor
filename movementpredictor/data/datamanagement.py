
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


def getTrackedBaseData(path, dim_x, dim_y, num_batch) -> list[TrackedObjectPosition]:

    try:
        extracted_tracks = [] 
        downsampled_frames = {}

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

                    frame = proto.frame
                    frame_data = frame.frame_data_jpeg
                    np_arr = np.frombuffer(frame_data, np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)  

                    resized_img = cv2.resize(img, (dim_x, dim_y), interpolation=cv2.INTER_AREA)
                    tensor_img = torch.from_numpy(resized_img).float() / 255.0
                    downsampled_frames[frame.timestamp_utc_ms] = tensor_img

                    detections = proto.detections

                    for detection in detections:

                        if detection.class_id == 2: # only cars for now

                            track = TrackedObjectPosition()
                            track.set_capture_ts(frame.timestamp_utc_ms)
                            track.set_uuid(detection.object_id)
                            track.set_class_id(detection.class_id)
                            bbox = detection.bounding_box
                            track.set_center([bbox.min_x + (bbox.max_x-bbox.min_x)/2, bbox.min_y + (bbox.max_y-bbox.min_y)/2])
                            track.set_bbox([[bbox.min_x, bbox.min_y], [bbox.max_x, bbox.max_y]])
                            extracted_tracks.append(track)

                if count >= (num_batch+1)*100000:
                    break

        return downsampled_frames, extracted_tracks

    except Exception as e:

        print(f"Error processing the file: {e}")
        return None
