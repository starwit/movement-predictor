import base64
from pydantic import BaseModel
from typing import Optional, List


class TrackedObjectPosition(BaseModel):
    capture_ts: Optional[int] = None
    uuid: Optional[str] = None
    class_id: Optional[int] = None
    center: Optional[List[float]] = None  # [x, y]
    bbox: Optional[List[List[float]]] = None  # [[x1, y1], [x2, y2]]
    movement_angle: Optional[float] = None

    def set_class_id(self, class_id: int):
        self.class_id = class_id

    def set_capture_ts(self, capture_ts: int):
        self.capture_ts = capture_ts

    def set_uuid(self, uuid: str):
        self.uuid = uuid

    def set_bbox(self, bbox: List[List[float]]):
        """bbox: [[x1, y1], [x2, y2]]"""
        if len(bbox) != 2 or len(bbox[0]) != 2 or len(bbox[1]) != 2:
            raise ValueError("Center must be a list of this structure [[x1, y1], [x2, y2]]")
        self.bbox = bbox
    
    def set_center(self, center: List[float]):
        """center: [x, y]"""
        if len(center) != 2:
            raise ValueError("Center must be a list with exactly two float values [x, y].")
        self.center = center
    
    def set_movement_angle(self, angle: float):
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



