import logging
from math import atan, degrees
from typing import Dict
from movementpredictor.data.trackedobjectposition import TrackedObjectPosition
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import TheilSenRegressor
from scipy.signal import medfilt


class DataFilterer():
    log = logging.getLogger(__name__)
    min_movement = 0.05
    min_length = 7
    time_window_movement = 10000

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
        mapping = self._extract_vehicles_with_movement2(mapping)
        mapping = self._smooth_trajectories_add_movement_angle(mapping)

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
        mapping = self._smooth_trajectories_add_movement_angle(mapping)

        return mapping

    def _smooth_trajectories_add_movement_angle(self, mapping):
        new_mapping = {}
        for key, tracks_of_object in tqdm(mapping.items(), desc="smooth tracks"):
            if len(tracks_of_object) < self.min_length:
                continue

            bboxes = [track.get_bbox() for track in tracks_of_object]
            smooth_bboxes, smooth_centers = DataFilterer._smooth_trajectory_median(bboxes)#, timestamps)

            for track, bbox, center in zip(tracks_of_object, smooth_bboxes, smooth_centers):
                track.set_bbox(bbox)
                track.set_center(center)

            DataFilterer._calculate_movement_angle(tracks_of_object)
            new_mapping[key] = tracks_of_object
            
        return new_mapping

    def _extract_vehicles_with_movement(self, mapping):
        new_mapping = {}

        for key, tracks_of_object in mapping.items():
            min_x = min(tracks_of_object, key=lambda obj: obj.get_center()[0]).get_center()[0]
            max_x = max(tracks_of_object, key=lambda obj: obj.get_center()[0]).get_center()[0]
            if max_x - min_x < self.min_movement:
                min_y = min(tracks_of_object, key=lambda obj: obj.get_center()[1]).get_center()[1]
                max_y = max(tracks_of_object, key=lambda obj: obj.get_center()[1]).get_center()[1]
                if max_y - min_y < self.min_movement:
                    continue
            new_mapping[key] = tracks_of_object
        
        return new_mapping
    
    def _extract_vehicles_with_movement2(self, mapping):
        self.log.info("filter out parts without movement")
        new_mapping = {}
        for key, tracks_of_object in mapping.items():

            if len(tracks_of_object) < self.min_length:
                continue
            
            # skip trajectories with no movement at all
            min_x = min(tracks_of_object, key=lambda obj: obj.get_center()[0]).get_center()[0]
            max_x = max(tracks_of_object, key=lambda obj: obj.get_center()[0]).get_center()[0]
            total_movement = np.linalg.norm(min_x - max_x)

            if total_movement < self.min_movement:
                continue
            
            # sort out parts without movement
            timestamps = np.array([track.get_capture_ts() for track in tracks_of_object])
            traj = np.array([track.get_center() for track in tracks_of_object])
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
    def _smooth_trajectory_median(bboxes, kernel_size=3):
        bboxes = np.array(bboxes)
    
        # Eckpunkte extrahieren
        x_min = bboxes[:, 0, 0]
        y_min = bboxes[:, 0, 1]
        x_max = bboxes[:, 1, 0]
        y_max = bboxes[:, 1, 1]

        # Median-Filter auf alle Koordinaten anwenden
        x_min_smooth = medfilt(x_min, kernel_size)
        y_min_smooth = medfilt(y_min, kernel_size)
        x_max_smooth = medfilt(x_max, kernel_size)
        y_max_smooth = medfilt(y_max, kernel_size)

        # Neue Bounding Boxes zusammensetzen
        smoothed_bboxes = np.stack([
            np.stack([x_min_smooth, y_min_smooth], axis=1),
            np.stack([x_max_smooth, y_max_smooth], axis=1)
        ], axis=1)

        cx_smooth = (x_min_smooth + x_max_smooth) / 2
        cy_smooth = (y_min_smooth + y_max_smooth) / 2
        smoothed_centers = np.stack([cx_smooth, cy_smooth], axis=1)

        return smoothed_bboxes, smoothed_centers
    

    @staticmethod
    def _calculate_movement_angle(tracks_of_object: list[TrackedObjectPosition]):

        for i in range(len(tracks_of_object) - 6):
            tracks = tracks_of_object[i:i+7]
            x = np.array([track.get_center()[0] for track in tracks]).reshape(-1, 1)
            y = np.array([track.get_center()[1] for track in tracks]).ravel()

            if np.linalg.norm(np.array(tracks[-1].get_center()) - np.array(tracks[0].get_center())) < 0.015 and i > 0:
                angle = tracks_of_object[i-1].get_movement_angle()

            else:
                model = TheilSenRegressor()
                model.fit(x, y)
                slope = model.coef_[0]
                intercept = model.intercept_
                x_min = np.min(x)
                point1 = [x_min, slope * x_min + intercept]
                point2 = [x_min + 0.5, slope * (x_min + 0.5) + intercept]

                prev_track, next_track = TrackedObjectPosition(), TrackedObjectPosition()
                prev_track.set_center(point1)
                next_track.set_center(point2)
                angle = DataFilterer._get_angle(next_track, prev_track)

            tracks_of_object[i+3].set_movement_angle(angle)

            if i == 0:
                tracks_of_object[i].set_movement_angle(angle)
                tracks_of_object[i+1].set_movement_angle(angle)
                tracks_of_object[i+2].set_movement_angle(angle)
                
            if i == len(tracks_of_object) - 7:
                tracks_of_object[i+4].set_movement_angle(angle)
                tracks_of_object[i+5].set_movement_angle(angle)
                tracks_of_object[i+6].set_movement_angle(angle)
    

    @staticmethod
    def _get_angle(track, previous_track) -> float:
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


__all__ = ["DataFilterer"]