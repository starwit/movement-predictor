import logging
import os
from dotenv import load_dotenv

class ModelConfig:
    log = logging.getLogger(__name__)
    instance = None

    def __init__(self):
        load_dotenv()

        self.camera = os.getenv("CAMERA")
        self.name_data = os.getenv("NAME_DATA")
        self.name_model = os.getenv("NAME_MODEL")
        self.path_sae_data = os.getenv("PATH_SAE_DATA")
        self.dim_x = int(os.getenv("DIM_X"))
        self.dim_y = int(os.getenv("DIM_Y"))
        self.percentage_anomaly = float(os.getenv("PERCENTAGE_OF_ANOMALIES"))

        self.path_model = os.path.join("models", self.camera, self.name_model)
        self.path_store_data = os.path.join("movementpredictor/data/datasets", self.camera, self.name_data)

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = ModelConfig()
        return cls.instance