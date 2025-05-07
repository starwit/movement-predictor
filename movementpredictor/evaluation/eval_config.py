import logging
import os
from dotenv import load_dotenv

class EvalConfig:
    log = logging.getLogger(__name__)
    instance = None

    def __init__(self):
        load_dotenv()

        self.path_test_data = os.getenv("PATH_TEST_DATA")
        streamname = os.path.splitext(os.path.basename(self.path_test_data))[0]
        self.path_label_box = os.getenv("PATH_LABEL_BOX")
        self.path_sae_dumps = os.getenv("PATH_SAE_DUMPS", "").split(",")
        self.num_anomalies = float(os.getenv("NUM_ANOMALIES"))
        self.camera = os.getenv("CAMERA")

        self.path_store_anomalies = os.path.join(os.getenv("PATH_STORE_PREDICTED_ANOMALIES"), self.camera, streamname)
    
        self.model_architecture = os.getenv("MODEL_ARCHITECTURE", "MobileNet_v3")      
        self.output_distribution = os.getenv("OUTPUT_DISTR", "symmetric")
        self.path_model = os.getenv("MODEL_WEIGHTS")

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = EvalConfig()
        return cls.instance