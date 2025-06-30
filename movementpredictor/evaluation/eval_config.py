import logging
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path


class EvalConfig:
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

        self.path_test_data = os.getenv("PATH_TEST_DATA")
        streamname = os.path.splitext(os.path.basename(self.path_test_data))[0]
        self.path_label_box = os.getenv("PATH_LABEL_BOX")
        self.path_sae_dump = os.getenv("PATH_SAE_DUMP", "")
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