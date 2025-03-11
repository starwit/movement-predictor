import logging
from math import atan, degrees
from typing import Dict
from movementpredictor.data.trackedobjectposition import TrackedObjectPosition
from tqdm import tqdm
import numpy as np
from filterpy.kalman import KalmanFilter
from sklearn.linear_model import TheilSenRegressor
from scipy.signal import medfilt


class DataFilterer:
    log = logging.getLogger(__name__)
    max_angle_change = 50
    max_millisec_between_3_detections = 2000
    min_length = 7

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

        last_mapping = {}
        for key, tracks_of_object in tqdm(new_mapping.items(), desc="filtering tracks"):
            if len(tracks_of_object) < self.min_length:
                continue

            bboxes = [track.get_bbox() for track in tracks_of_object]
            timestamps = [track.get_capture_ts() for track in tracks_of_object]

            #seperate_indices = DataFilterer.separate_timestamps(timestamps)
            smooth_bboxes, smooth_centers = DataFilterer.smooth_trajectory_median(bboxes)#, timestamps)

            for track, bbox, center in zip(tracks_of_object, smooth_bboxes, smooth_centers):
                track.set_bbox(bbox)
                track.set_center(center)

            DataFilterer.calculate_movement_angle(tracks_of_object)
            last_mapping[key] = tracks_of_object
            
        return last_mapping
    

    @staticmethod
    def smooth_trajectory_median(bboxes, kernel_size=3):
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
    def smooth_trajectory(boxes, timestamps):
        # smooth center, width, higth of bbox
        boxes = np.array(boxes)
        cx = (boxes[:, 0, 0] + boxes[:, 1, 0]) / 2
        cy = (boxes[:, 0, 1] + boxes[:, 1, 1]) / 2
        w = boxes[:, 1, 0] - boxes[:, 0, 0]
        h = boxes[:, 1, 1] - boxes[:, 0, 1]

        # Kalman-Filter für cx, cy, w, h
        kf_cx = DataFilterer.create_kalman_filter()
        kf_cy = DataFilterer.create_kalman_filter()
        kf_w = DataFilterer.create_kalman_filter()
        kf_h = DataFilterer.create_kalman_filter()

        # Glättung der Bounding Boxes
        cx_smooth = np.zeros_like(cx)
        cy_smooth = np.zeros_like(cy)
        w_smooth = np.zeros_like(w)
        h_smooth = np.zeros_like(h)

        for i in range(len(timestamps)):
            dt = timestamps[i] - timestamps[i - 1] if i > 0 else 0

            # Update Kalman-Filter für cx
            kf_cx.F = np.array([[1, dt], [0, 1]])  # Zustandsübergangsmatrix
            kf_cx.predict()
            kf_cx.update(cx[i])
            cx_smooth[i] = kf_cx.x[0]

            # Update Kalman-Filter für cy
            kf_cy.F = np.array([[1, dt], [0, 1]])  # Zustandsübergangsmatrix
            kf_cy.predict()
            kf_cy.update(cy[i])
            cy_smooth[i] = kf_cy.x[0]

            # Update Kalman-Filter für w
            kf_w.F = np.array([[1, dt], [0, 1]])  # Zustandsübergangsmatrix
            kf_w.predict()
            kf_w.update(w[i])
            w_smooth[i] = kf_w.x[0]

            # Update Kalman-Filter für h
            kf_h.F = np.array([[1, dt], [0, 1]])  # Zustandsübergangsmatrix
            kf_h.predict()
            kf_h.update(h[i])
            h_smooth[i] = kf_h.x[0]

        # Neue Bounding Boxes
        x_min = cx_smooth - w_smooth / 2
        x_max = cx_smooth + w_smooth / 2
        y_min = cy_smooth - h_smooth / 2
        y_max = cy_smooth + h_smooth / 2

        new_bboxes = np.stack([np.stack([x_min, y_min], axis=1), np.stack([x_max, y_max], axis=1)], axis=1)
        new_centers = np.stack([cx_smooth, cy_smooth], axis=1)
        return new_bboxes, new_centers

    
    @staticmethod
    def create_kalman_filter():
        """
        Klaman-filter for [position, velocity].
        """
        #kf = KalmanFilter(dim_x=2, dim_z=1)
        #kf.x = np.array([0, 0])  # Initialzustand [position, velocity]
        #kf.P = np.eye(2) * 1000  # Anfängliche Unsicherheit
        #kf.R = 10  # Messunsicherheit
        #kf.Q = np.array([[1, 0], [0, 1]])  # Prozessrauschen
        #kf.H = np.array([[1, 0]])  # Messmatrix

        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([0, 0])  # Initialzustand [position, velocity]
        kf.P = np.eye(2) * 10000  # Anfangsunsicherheit
        kf.R = 100  # Erhöhte Messunsicherheit -> stärkeres Glätten
        kf.Q = np.array([[0.1, 0], [0, 0.1]])  # Geringeres Prozessrauschen
        kf.H = np.array([[1, 0]])  # Messmatrix
        return kf

    
    def apply_filtering_(self, tracking_list: list[TrackedObjectPosition]) -> Dict[str, list[TrackedObjectPosition]]:
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
            skip = 0
                
            for i in range(len(tracks_of_object) - 2):
                if skip > 0:
                    skip -= 1
                    continue

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
                    #speed_change = DataFilterer.get_speed_diff(track, prev_track, prev_prev_track)
                    
                    if angle_change < DataFilterer.max_angle_change:
                        trajectory_angle = DataFilterer.get_angle(track, prev_prev_track)
                        if trajectory_angle == -1:
                            skip = 2
                            continue
                        if prev_prev_track not in updated_tracks:
                            updated_tracks.append(prev_prev_track)
                        if prev_track not in updated_tracks:
                            updated_tracks.append(prev_track)
                        if track not in updated_tracks:
                            updated_tracks.append(track)
                    else: 
                        skip = 2
                else: 
                    break

            DataFilterer.calculate_movement_angle(updated_tracks)
            new_mapping[key] = updated_tracks  

            for track in updated_tracks:
                if track.get_movement_angle() is None:
                    print("ALARRRRRM!")

        return new_mapping
    

    @staticmethod
    def calculate_movement_angle(tracks_of_object: list[TrackedObjectPosition]):

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
                angle = DataFilterer.get_angle(next_track, prev_track)

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
    def calculate_movement_angle_(tracks_of_object):
        for i in range(len(tracks_of_object) - 2):
                prev_track = tracks_of_object[i]
                track = tracks_of_object[i + 1]
                next_track = tracks_of_object[i + 2]

                success = False
                j = i-1
                while j > max(-1, i-10):
                    if np.linalg.norm(np.array(next_track.get_center()) - np.array(prev_track.get_center())) < 0.015:
                        prev_track = tracks_of_object[j]
                    else:
                        success = True
                        break
                    j -= 1
                    
                if not success and j > -1:
                    angle = tracks_of_object[j].get_movement_angle()
                else:
                    angle = DataFilterer.get_angle(next_track, prev_track)
                
                track.set_movement_angle(angle)
                if prev_track.get_movement_angle() is None:
                    prev_track.set_movement_angle(angle)
                if i == len(tracks_of_object) - 3:
                    next_track.set_movement_angle(angle)

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
