import base64

class TrackedObjectPosition:

    def __init__(self):
        self.capture_ts = None
        self.uuid = None
        self.class_id = None
        self.center = None
        self.bbox = None
        self.movement_angle = None

    def set_class_id(self, class_id):
        self.class_id = class_id

    def set_capture_ts(self, capture_ts):
        self.capture_ts = capture_ts

    def set_uuid(self, uuid):
        self.uuid = uuid

    def set_bbox(self, bbox):
        self.bbox = bbox
    
    def set_center(self, center):
        self.center = center
    
    def set_movement_angle(self, angle):
        self.movement_angle = angle

    def get_class_id(self):
        return self.class_id

    def get_capture_ts(self):
        return self.capture_ts

    def get_bbox(self):
        return self.bbox

    def get_uuid(self):
        return self.uuid
    
    def get_center(self):
        return self.center
    
    def get_movement_angle(self):
        return self.movement_angle

    
    def to_json(self):
        json_obj = {
            "captureTs": self.capture_ts.isoformat(),
            "uuid": base64.b64encode(self.uuid).decode('utf-8'),
            "classId": self.class_id,
            "center": self.center,
            "bbox": self.bbox,
            "frame_idx": self.frame_idx
        }
        return json_obj


