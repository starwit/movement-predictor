import base64

class TrackedObjectPosition:

    def __init__(self):
        self.capture_ts = None
        self.uuid = None
        self.class_id = None
        self.center = None
        self.trajectory_angle = None
        self.angle_look_up = None
        self.frame_ts = None


    def set_class_id(self, class_id):
        self.class_id = class_id

    def set_capture_ts(self, capture_ts):
        self.capture_ts = capture_ts

    def set_uuid(self, uuid):
        self.uuid = uuid

    def set_center(self, center):
        self.center = center

    def set_trajectory_angle(self, trajectory_angle):
        self.trajectory_angle = trajectory_angle

    def set_angle_look_up(self, angle_look_up):
        self.angle_look_up = angle_look_up

    def set_frame_ts(self, frame_ts):
        self.frame_ts = frame_ts


    def get_class_id(self):
        return self.class_id

    def get_capture_ts(self):
        return self.capture_ts

    def get_center(self):
        return self.center

    def get_uuid(self):
        return self.uuid

    def get_trajectory_angle(self):
        return self.trajectory_angle

    def get_angle_look_up(self):
        return self.angle_look_up

    def get_frame_ts(self):
        return self.frame_ts

    
    def to_json(self):
        json_obj = {
            "captureTs": self.capture_ts.isoformat(),
            "uuid": base64.b64encode(self.uuid).decode('utf-8'),
            "classId": self.class_id,
            "center": self.center,
            "trajectoryAngle": self.trajectory_angle,
            "angleLookUp": self.angle_look_up
        }
        return json_obj


