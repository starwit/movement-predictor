import logging
from math import atan, degrees
from typing import Dict
from movementpredictor.data.trackedobjectposition import TrackedObjectPosition
from tqdm import tqdm


class DataFilterer:
    log = logging.getLogger(__name__)
    max_angle_change = 60
    max_millisec_between_3_detections = 2000

    def apply_filtering(self, tracking_list: list[TrackedObjectPosition]) -> Dict[str, list[TrackedObjectPosition]]:
        self.log.debug("Start filtering tracks")
        mapping = {}

        for track in tracking_list:
            key = track.uuid
            if key not in mapping:
                mapping[key] = []
            mapping[key].append(track)
        
        new_mapping = {}
        for key, tracks_of_object in mapping.items():
            min_x = min(tracks_of_object, key=lambda obj: obj.get_center()[0]).get_center()[0]
            max_x = max(tracks_of_object, key=lambda obj: obj.get_center()[0]).get_center()[0]
            if max_x - min_x < 0.05:
                min_y = min(tracks_of_object, key=lambda obj: obj.get_center()[1]).get_center()[1]
                max_y = max(tracks_of_object, key=lambda obj: obj.get_center()[1]).get_center()[1]
                if max_y - min_y < 0.05:
                    continue
            new_mapping[key] = tracks_of_object

        for key, tracks_of_object in tqdm(new_mapping.items(), desc="filtering tracks"):
            updated_tracks = []
            for i in range(len(tracks_of_object) - 2):
                prev_prev_track = tracks_of_object[i]
                prev_track = tracks_of_object[i + 1]
                track = tracks_of_object[i + 2]

                # remove tracks at the border
                #skip = False
                #for t in [prev_prev_track, prev_track, track]:
                #    if t.center[0] < 0.05 or t.center[0] > 0.95 or t.center[1] < 0.05 or t.center[1] > 0.95:
                #        skip = True
                #if skip:
                #    continue

                if track.capture_ts - prev_prev_track.capture_ts <= DataFilterer.max_millisec_between_3_detections:
                    angle_change = DataFilterer.get_angle_diff(track, prev_track, prev_prev_track)
                    if angle_change < DataFilterer.max_angle_change:
                        trajectory_angle = DataFilterer.get_angle(track, prev_prev_track)
                        if trajectory_angle == -1:
                            continue
                        if prev_prev_track not in updated_tracks:
                            updated_tracks.append(prev_prev_track)
                        if prev_track not in updated_tracks:
                            updated_tracks.append(prev_track)
                        if track not in updated_tracks:
                            updated_tracks.append(track)
                else: 
                    break
            new_mapping[key] = updated_tracks  

        return new_mapping


    @staticmethod
    def get_angle_diff(track, prev_track, prev_prev_track) -> float:
        angle = DataFilterer.get_angle(track, prev_track)
        next_angle = DataFilterer.get_angle(prev_track, prev_prev_track)
        if angle == -1 or next_angle == -1:
            return 0
        change = abs(next_angle - angle)
        return min(change, 360. - change)

    @staticmethod
    def get_angle(track, previous_track) -> float:
        delta_x = track.center[0] - previous_track.center[0]
        delta_y = (track.center[1] - previous_track.center[1]) * (-1)
        if delta_x == 0 or delta_y == 0:
            return -1

        angle = degrees(atan(abs(delta_y) / abs(delta_x)))
        if delta_x < 0 and delta_y >= 0:
            angle = 180. - angle
        elif delta_x >= 0 and delta_y < 0:
            angle = 360. - angle
        elif delta_x < 0 and delta_y < 0:
            angle = 180. + angle
        return angle
