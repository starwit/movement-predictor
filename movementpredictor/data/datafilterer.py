import logging
from math import atan, degrees
from typing import Dict
from movementpredictor.data.datamanagement import TrackedObjectPosition
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import TheilSenRegressor
from scipy.signal import medfilt


class DataFilterer():

    log = logging.getLogger(__name__)
    min_movement = 0.05
    min_length = 7
    time_window_movement = 10000        # 10s
    max_angle_change = 60


    def apply_filtering(self, tracking_list: list[TrackedObjectPosition]) -> Dict[str, list[TrackedObjectPosition]]:
        """
        Sort out tracks of vehicles that have not been moving for a certain time and smooth trajectory

        Args:
            tracking_list: a list containing single tracks of type TrackedObjectPosition, on which anomaly detection should be performed later
        
        Returns:
            dict: the adapted tracks are grouped together to trajectories -> key: object id, value: list[TrackedObjectPosition] which is this object's trajectory
        """
        self.log.info("Start filtering tracks")

        mapping = DataFilterer._map_tracks_to_id(tracking_list)
        mapping = self._extract_vehicles_with_movement(mapping)
        mapping = self._smooth_trajectories_add_movement_info(mapping)

        return mapping
    
    
    def only_smoothing(self, tracking_list: list[TrackedObjectPosition]) -> Dict[str, list[TrackedObjectPosition]]:
        """
        Smoothes trajectories without previous filtering. Should only be used for small tests not when performing the real anomaly detection!

        Args:
            tracking_list: a list containing single tracks of type TrackedObjectPosition
        
        Returns:
            dict: the adapted tracks are grouped together to trajectories -> key: object id, value: list[TrackedObjectPosition] which is this object's trajectory
        """
        mapping = DataFilterer._map_tracks_to_id(tracking_list)
        mapping = self._smooth_trajectories_add_movement_info(mapping)

        return mapping
    

    def _smooth_trajectories_add_movement_info(self, mapping):
        new_mapping = {}
        for key, tracks_of_object in tqdm(mapping.items(), desc="smooth tracks"):
            if len(tracks_of_object) < self.min_length:
                continue

            bboxes = [track.bbox for track in tracks_of_object]
            #timestamps = [track.capture_ts for track in tracks_of_object]
            smooth_bboxes, smooth_centers = DataFilterer._smooth_trajectory_median(bboxes)#, timestamps)
            #smooth_bboxes, smooth_centers = DataFilterer._smooth_trajectory_median_(bboxes, timestamps)

            for track, bbox, center in zip(tracks_of_object, smooth_bboxes, smooth_centers):
                track.bbox = bbox
                track.center = center

            self._calculate_movement_angle(tracks_of_object)
            #DataFilterer._calculate_movement_speed(tracks_of_object)
            new_mapping[key] = tracks_of_object
            
        return new_mapping
    
    
    def _extract_vehicles_with_movement(self, mapping):
        self.log.info("filter out parts without movement")
        new_mapping = {}
        for key, tracks_of_object in mapping.items():

            if len(tracks_of_object) < self.min_length:
                continue
            
            # skip trajectories with no movement at all
            min_x = min(tracks_of_object, key=lambda obj: obj.center[0]).center[0]
            max_x = max(tracks_of_object, key=lambda obj: obj.center[0]).center[0]
            total_movement = np.linalg.norm(min_x - max_x)

            if total_movement < self.min_movement:
                continue
            
            # sort out parts without movement
            timestamps = np.array([track.capture_ts for track in tracks_of_object])
            traj = np.array([track.center for track in tracks_of_object])
            keep = np.zeros(len(traj), dtype=bool)

            for i, (timestamp, point) in enumerate(zip(timestamps, traj)):
                start = np.searchsorted(timestamps, timestamp - self.time_window_movement, side='left')
                end = np.searchsorted(timestamps, timestamp + self.time_window_movement, side='right')
                
                neighbors = traj[start:end]
                distances = np.linalg.norm(neighbors - point, axis=1)
                if np.any(distances >= self.min_movement):
                    keep[i] = True
            
            # Filter trajectory
            filtered_traj = [tracks_of_object[i] for i in range(len(tracks_of_object)) if keep[i]]
            new_mapping[key] = filtered_traj
        
        return new_mapping
    

    def _calculate_movement_angle(self, tracks_of_object: list[TrackedObjectPosition]):
        i = 0
        while i < len(tracks_of_object) - 4:
            tracks = tracks_of_object[i:i+5]

            change1 = DataFilterer._get_angle_diff(tracks[3], tracks[2], tracks[1])
            change2 = DataFilterer._get_angle_diff(tracks[4], tracks[2], tracks[0])
            change3 = DataFilterer._get_angle_diff(tracks[4], tracks[2], tracks[1])
            change4 = DataFilterer._get_angle_diff(tracks[3], tracks[2], tracks[0])

            if all(change < self.max_angle_change for change in [change1, change2, change3, change4]):
                tracks_of_object[i+2].clear_detection = True
                if i == 0:
                    tracks_of_object[0].clear_detection = True
                    tracks_of_object[1].clear_detection = True
                if i == len(tracks_of_object) - 5:
                    tracks_of_object[-1].clear_detection = True
                    tracks_of_object[-2].clear_detection = True
                i += 1
            else:
                i += 3
        
        for i in range(len(tracks_of_object) - 6):
            tracks = tracks_of_object[i:i+7]
            tracks = [t for t in tracks if t.clear_detection]
            angle = None

            if (len(tracks) < 2 or np.linalg.norm(np.array(tracks[-1].center) - np.array(tracks[0].center)) < 0.015) and i > 0:
                angle = tracks_of_object[i-1].movement_angle

            elif len(tracks) >= 2:   
                angle = DataFilterer._calculate_smooth_angle(tracks)

            if angle is None:
                    angle = DataFilterer._calculate_smooth_angle(tracks_of_object[i:i+7])

            tracks_of_object[i+3].movement_angle = angle

            if i == 0:
                for j in range(3):
                    tracks_of_object[i + j].movement_angle = angle
            if i == len(tracks_of_object) - 7:
                for j in range(4, 7):
                    tracks_of_object[i + j].movement_angle = angle

    
    @staticmethod
    def _calculate_smooth_angle(tracks):
        x = np.array([track.center[0] for track in tracks]).reshape(-1, 1)
        y = np.array([track.center[1] for track in tracks]).ravel()

        model = TheilSenRegressor()
        model.fit(x, y)
        slope = model.coef_[0]
        intercept = model.intercept_

        point1 = [x[0], slope * x[0] + intercept]
        point2 = [x[-1], slope * x[-1] + intercept]

        return DataFilterer._get_angle(point2, point1)


    @staticmethod
    def _map_tracks_to_id(tracking_list: list[TrackedObjectPosition]):
        mapping = {}

        for track in tracking_list:
            key = track.uuid
            if key not in mapping:
                mapping[key] = []
            mapping[key].append(track)

        return mapping
    

    @staticmethod
    def _smooth_trajectory_median(bboxes, kernel_size=7):
        bboxes = np.array(bboxes)

        x_min = bboxes[:, 0, 0]
        y_min = bboxes[:, 0, 1]
        x_max = bboxes[:, 1, 0]
        y_max = bboxes[:, 1, 1]

        x_min_smooth = medfilt(x_min, kernel_size)
        y_min_smooth = medfilt(y_min, kernel_size)
        x_max_smooth = medfilt(x_max, kernel_size)
        y_max_smooth = medfilt(y_max, kernel_size)

        x_min[kernel_size//2:-kernel_size//2] = x_min_smooth[kernel_size//2:-kernel_size//2]
        x_max[kernel_size//2:-kernel_size//2] = x_max_smooth[kernel_size//2:-kernel_size//2]
        y_min[kernel_size//2:-kernel_size//2] = y_min_smooth[kernel_size//2:-kernel_size//2]
        y_max[kernel_size//2:-kernel_size//2] = y_max_smooth[kernel_size//2:-kernel_size//2]

        smoothed_bboxes = np.stack([
            np.stack([x_min, y_min], axis=1),
            np.stack([x_max, y_max], axis=1)
        ], axis=1)

        cx_smooth = (x_min + x_max) / 2
        cy_smooth = (y_min + y_max) / 2
        smoothed_centers = np.stack([cx_smooth, cy_smooth], axis=1)

        return smoothed_bboxes, smoothed_centers
    

    @staticmethod
    def _calculate_movement_speed(tracks_of_object: list[TrackedObjectPosition]):

        for i in range(len(tracks_of_object)-1):
            track = tracks_of_object[i]
            next_track = tracks_of_object[i+1]

            time_diff = next_track.capture_ts - track.capture_ts
            distance = np.linalg.norm(next_track.center-track.center)
            speed = distance/(time_diff/1000)           # distance in frame image per second

            track.movement_speed = speed
            if i == len(tracks_of_object)-2:
                tracks_of_object[-1].movement_speed = speed

    
    @staticmethod
    def _get_angle_diff(track: TrackedObjectPosition, prev_track: TrackedObjectPosition, prev_prev_track: TrackedObjectPosition) -> float:
        angle1 = DataFilterer._get_angle(track.center, prev_track.center)
        angle2 = DataFilterer._get_angle(prev_track.center, prev_prev_track.center)
        diff = abs(angle2 - angle1)
        return min(diff, 360 - diff)
    

    @staticmethod
    def _get_angle(center, previous_center) -> float:
        delta_x = center[0] - previous_center[0]
        delta_y = (center[1] - previous_center[1]) * (-1)
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


__all__ = ["DataFilterer"]