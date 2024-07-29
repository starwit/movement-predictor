import time
import logging
from datetime import datetime, timezone
from AEsAnomalyDetection.TrackedObjectPosition import TrackedObjectPosition


log = logging.getLogger(__name__)


class TimedTrajectories:
    def __init__(self, timeout):
        self.timeout = timeout
        self.data = {}
        self.timestamps = {}
        self.out = []

    def add(self, proto):
        current_time = time.time()

        #with open("output.txt", "w") as file:
         #   print(proto, file=file)
        frame = proto.frame
        last_tracks = []

        border_box_count = 0
        
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

            #scale = frame.shape.height / frame.shape.width
            #print(scale)
            center = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5) #* scale)
            #print(center)

            tracked_object_pos = TrackedObjectPosition()
            #tracked_object_pos.set_capture_ts(frame.timestamp_utc_ms)
            datetime_obj = datetime.fromtimestamp(frame.timestamp_utc_ms/1_000, tz=timezone.utc)
            tracked_object_pos.set_capture_ts(datetime_obj)
            tracked_object_pos.set_uuid(det.object_id)
            tracked_object_pos.set_class_id(det.class_id)
            tracked_object_pos.set_center(center)
            #print(tracked_object_pos.get_capture_ts())
            #print(tracked_object_pos.get_center())
            #print(tracked_object_pos.get_uuid())

            last_tracks.append(tracked_object_pos)

        log.debug(f"Removed {border_box_count} detections on frame border.")

        for track in last_tracks:
            id = track.get_uuid()
            if id not in self.data:
                self.data[id] = []
            self.data[id].append(track)
            if not id in self.timestamps.keys():
                self.timestamps[id] = [current_time, current_time]      # first entry: first seen, second entry: last_seen
            else:
                self.timestamps[id][1] = current_time
            self._check_time()

    def _check_time(self):
        ready = []
        for id in self.timestamps.keys():
            time_diff = self.timestamps[id][1] - self.timestamps[id][0]
            if time_diff > 10:
                ready.append(self.data[id])
                self.timestamps[id][0] +=  1
                new_data = []
                for track in self.data[id]:
                    if track.get_capture_ts().timestamp() > self.timestamps[id][0]:
                        new_data.append(track)
                self.data[id] = new_data

        current_time = time.time()
        expired_keys = [id for id, [_, ts2] in self.timestamps.items() if current_time - ts2 >= self.timeout]
        for id in expired_keys:
            self.timestamps.pop(id)        
            self.data.pop(id)
        
        #print(self.out)
        #print(ready)
        self.out = self.out + ready
        #print(self.out)


    def get_latest_Trajectories(self):
        track_list = self.out.copy()
        self.out = []
        return track_list

    
    def calc_frame_border_threshold(self, frame):
        width_height = frame.shape.width / frame.shape.height
        return [0.01, width_height * 0.01]
