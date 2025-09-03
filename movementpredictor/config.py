import logging
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path


def str_to_bool(v: str) -> bool:
    return v.lower() in ("true", "1", "yes", "y", "on")


class ModelConfig:
    log = logging.getLogger(__name__)
    instance = None

    def __init__(self):
        env_path = find_dotenv(raise_error_if_not_found=False)

        if not env_path:
            tpl = Path(__file__).parent / ".env.template"
            if tpl.exists():
                env_path = str(tpl)
            else:
                self.log.error("Could not find .env or .env.template.")
                exit(1)

        load_dotenv(env_path)

        raw_steps =  os.getenv("COMPUTE_STEPS", "prepare, train, threshold")
        self.steps = [p.strip() for p in raw_steps.split(",") if p.strip()]
        self.visualize = str_to_bool(os.getenv("VISUALIZE", "False"))

        self.camera = os.getenv("CAMERA")
        self.pixel_per_axis = int(os.getenv("PIXEL_PER_AXIS"))
        self.frame_rate = float(os.getenv("FRAME_RATE", "10"))
        self.time_diff_prediction = float(os.getenv("TIME_DIFF_PREDICTION"))

        self.name_data = os.getenv("NAME_DATA")
        self.path_sae_data_train = os.getenv("PATH_SAE_DATA_TRAIN")
        raw_paths_sae_data_test = os.getenv("PATHS_SAE_DATA_TEST")
        self.paths_sae_data_test = [p.strip() for p in raw_paths_sae_data_test.split(",") if p.strip()]

        for path_str in self.paths_sae_data_test:
            p = Path(path_str)
            if not p.exists():
                self.log.error("did not find sae-dump: " + str(path_str))

        self.path_store_data = os.path.join("movementpredictor/data/datasets", self.camera, self.name_data)
        os.makedirs(self.path_store_data, exist_ok=True)

        self.model_architecture = os.getenv("MODEL_ARCHITECTURE", "MobileNet_v3")      
        self.output_distribution = os.getenv("OUTPUT_DISTR", "symmetric")
        self.class_of_interest = int(os.getenv("YOLO_OBJECT_TYPE_OF_INTEREST", "2"))

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