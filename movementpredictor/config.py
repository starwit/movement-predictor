import logging
import os
from dotenv import load_dotenv

class ModelConfig:
    log = logging.getLogger(__name__)
    instance = None

    def __init__(self):
        load_dotenv()

        self.start_time = os.getenv("START_TIME")
        self.end_time = os.getenv("END_TIME")
        self.database_url = os.getenv("DATABASE_URL")       # IP 100.70.113.34 = infra-brain01
        self.user = os.getenv("DATABASE_USER")
        self.password = os.getenv("DATABASE_PASSWORD")
        self.ssl = os.getenv("DATABASE_SSL")
        self.database_table = os.getenv("DATABASE_TABLE")
        self.camera_id = os.getenv("DATABASE_CAMERA_ID")
        self.path_model = os.getenv("PATH_MODEL")
        self.path_sae_data = os.getenv("PATH_SAE_DATA")

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = ModelConfig()
        return cls.instance