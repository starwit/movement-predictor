import logging
import os
from dotenv import load_dotenv

class ModelConfig:
    log = logging.getLogger(__name__)
    instance = None

    def __init__(self):
        load_dotenv()

        self.camera = os.getenv("CAMERA")
        self.pixel_per_axis = int(os.getenv("PIXEL_PER_AXIS"))
        self.time_diff_prediction = float(os.getenv("TIME_DIFF_PREDICTION"))

        self.name_data = os.getenv("NAME_DATA")
        self.path_sae_data_train = os.getenv("PATH_SAE_DATA_TRAIN")
        self.path_sae_data_test = os.getenv("PATH_SAE_DATA_TEST")
        self.path_store_data = os.path.join("movementpredictor/data/datasets", self.camera, self.name_data)
        os.makedirs(self.path_store_data, exist_ok=True)

        self.model_architecture = os.getenv("MODEL_ARCHITECTURE", "MobileNet_v3")      
        self.output_distribution = os.getenv("OUTPUT_DISTR", "symmetric")
        self.name_model = self.model_architecture + "_" + self.output_distribution + "_prob"
        self.path_model = os.path.join("models", self.camera, self.name_data, self.name_model)

        self.path_plots = os.path.join("plots", self.camera, self.name_data, self.name_model)
        os.makedirs(self.path_plots, exist_ok=True)
        self.percentage_anomaly = float(os.getenv("PERCENTAGE_OF_ANOMALIES"))

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = ModelConfig()
        return cls.instance