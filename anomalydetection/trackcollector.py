import time
import logging
from datetime import datetime, timezone
from movementpredictor.data.trackedobjectposition import TrackedObjectPosition



class TrackCollector:
    border_threshold = None

    def __init__(self, log_level):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(log_level)
        self.tracks = []
        self.frames = []
        self.out_time_in_ms = 10000

    def add(self, proto):
        frame = proto.frame
        self.frames.append(frame)

        border_box_count = 0
        
        if self.border_threshold is None:
            self.border_threshold = self.calc_frame_border_threshold(frame)
        outer_threshold_x = 1 - self.border_threshold[0]
        outer_threshold_y = 1 - self.border_threshold[1]

        for det in proto.detections:
            if det.class_id != 2:
                continue

            min_x = det.bounding_box.min_x
            min_y = det.bounding_box.min_y
            max_x = det.bounding_box.max_x
            max_y = det.bounding_box.max_y

            # Remove tracks at border
            if min_x <= self.border_threshold[0] or min_y <= self.border_threshold[1] or max_x >= outer_threshold_x or max_y >= outer_threshold_y:
                border_box_count += 1
                continue

            center = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5) 

            tracked_object_pos = TrackedObjectPosition()
            tracked_object_pos.set_capture_ts(frame.timestamp_utc_ms)
            tracked_object_pos.set_uuid(det.object_id)
            tracked_object_pos.set_class_id(det.class_id)
            tracked_object_pos.set_center(center)
            tracked_object_pos.set_bbox([[min_x, min_y], [max_x, max_y]])

            self.tracks.append(tracked_object_pos)

        #self.log.debug(f"Removed {border_box_count} detections on frame border.")


    def get_latest_data(self):
        if self.frames[-1].timestamp_utc_ms - self.frames[0].timestamp_utc_ms < self.out_time_in_ms:
            return [], []
        
        track_list = self.tracks.copy()
        frame_list = self.frames.copy()

        self.frames = self.frames[-5:]
        timestamps = [frame.timestamp_utc_ms for frame in self.frames]
        self.tracks = [track for track in self.tracks if track.get_capture_ts() in timestamps]

        return track_list, frame_list

    
    def calc_frame_border_threshold(self, frame):
        width_height = frame.shape.width / frame.shape.height
        return [0.01, width_height * 0.01]