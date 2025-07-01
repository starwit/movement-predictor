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


        self.camera = os.getenv("CAMERA")

        self.path_test_data = os.getenv("PATH_TEST_DATA")
        #streamname = os.path.splitext(os.path.basename(self.path_test_data))[0]
        self.path_store_anomalies = os.path.join(os.getenv("PATH_STORE_PREDICTED_ANOMALIES"), self.camera)#, streamname)
        self.path_model = os.getenv("MODEL_WEIGHTS")

        self.path_label_box = os.getenv("PATH_LABEL_BOX")
        raw_path_sae_dumps =  os.getenv("PATH_SAE_DUMPS", "")
        self.path_sae_dumps = [p.strip() for p in raw_path_sae_dumps.split(",") if p.strip()]

        for path_str in self.path_sae_dumps:
            p = Path(path_str)
            if not p.exists():
                self.log.error("did not find sae-dump: " + str(path_str))

        self.num_anomalies = float(os.getenv("NUM_ANOMALIES"))
        self.model_architecture = os.getenv("MODEL_ARCHITECTURE", "MobileNet_v3")      
        self.output_distribution = os.getenv("OUTPUT_DISTR", "symmetric")


    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = EvalConfig()
        return cls.instance