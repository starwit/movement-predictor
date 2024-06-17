import base64
import threading
import time
import logging
from de.starwit.visionapi.Messages import Detection, SaeMessage, VideoFrame
from datetime import datetime
from queue import Queue, Empty
import redis
from google.protobuf.message import DecodeError
import psycopg2
from psycopg2.extensions import AsIs
import json
from collections import namedtuple
from Config import AnomalyConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineConnector(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.running = False
        config = AnomalyConfig()
        self.jedis = redis.StrictRedis(host=config['redisHost'], port=config['redisPort'])
        self.stream_offset_by_id = {f"{config['redisInputStreamPrefix']}:{id}": "$" for id in config['redisStreamIds']}
        self.protos = Queue(maxsize=10000)  # !!! we need blocking queue

    def calc_frame_border_threshold(self, frame: VideoFrame):
        width_height = frame.shape.width / frame.shape.height
        return [0.01, width_height * 0.01]

    def run(self):
        self.running = True

        while self.running:
            result = None
            try:
                result = self.jedis.xread(self.stream_offset_by_id, count=5, block=2000)
            except redis.exceptions.ConnectionError as ex:
                logger.error("Could not read from Redis")
                logger.debug(ex)
                time.sleep(1)
                continue

            if not result:
                continue

            for stream_id, messages in result.items():
                for message in messages:
                    self.stream_offset_by_id[stream_id] = message['id']
                    proto_b64 = message['proto_data_b64']
                    try:
                        proto_data = base64.b64decode(proto_b64)
                        proto = SaeMessage.FromString(proto_data)
                        self.protos.put(proto)
                    except DecodeError as e:
                        logger.error(f"Error decoding proto from message. streamId={stream_id}")
                        logger.debug(e)

            time.sleep(0.1)

    def get_frames_and_tracks(self):
        last_frames = []
        last_tracks = []

        border_box_count = 0
        while not self.protos.empty():
            try:
                proto = self.protos.get_nowait()
            except Empty:
                logger.error("Proto queue empty")
                break

            frame = proto.frame
            last_frames.append(frame)

            border_threshold = self.calc_frame_border_threshold(frame)
            outer_threshold_x = 1 - border_threshold[0]
            outer_threshold_y = 1 - border_threshold[1]

            for det in proto.detections:
                if det.class_id != 2:
                    continue

                min_x = det.bounding_box.min_x
                min_y = det.bounding_box.min_y
                max_x = det.bounding_box.max_x
                max_y = det.bounding_box.max_y

                # Remove tracks at border
                if min_x <= border_threshold[0] or min_y <= border_threshold[1] or max_x >= outer_threshold_x or max_y >= outer_threshold_y:
                    border_box_count += 1
                    continue

                scale = frame.shape.height / frame.shape.width
                center = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5 * scale)

                tracked_object_pos = {
                    'center': center,
                    'frame_ts': str(frame.timestamp_utc_ms),
                    'capture_ts': datetime.fromtimestamp(frame.timestamp_utc_ms / 1000.0),
                    'class_id': det.class_id,
                    'uuid': det.object_id.hex()
                }

                last_tracks.append(tracked_object_pos)

        logger.info(f"Removed {border_box_count} detections on frame border.")

        LatestFramesAndTracks = namedtuple('LatestFramesAndTracks', ['frames', 'tracks'])
        return LatestFramesAndTracks(last_frames, last_tracks)

    def start(self):
        self.running = True
        super().start()

    def stop(self):
        self.running = False