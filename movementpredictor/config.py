import logging
import os
from dotenv import load_dotenv

class ModelConfig:
    log = logging.getLogger(__name__)
    instance = None

    def __init__(self):
        load_dotenv()

        self.path_model = os.getenv("PATH_INFERENCE_BUNDLE")
        self.path_sae_data = os.getenv("PATH_SAE_DATA")
        self.path_store_data = os.getenv("PATH_STORE_DATA")
        self.dim_x = int(os.getenv("DIM_X"))
        self.dim_y = int(os.getenv("DIM_Y"))
        self.percentage_anomaly = int(os.getenv("PERCENTAGE_OF_ANOMALIES"))

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = ModelConfig()
        return cls.instance