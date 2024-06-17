import logging
import os
from dotenv import load_dotenv

class AnomalyConfig:
    log = logging.getLogger(__name__)
    instance = None

    def __init__(self):
        load_dotenv()

        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', '6379'))
        self.redis_stream_ids = os.getenv('REDIS_STREAM_IDS', '').split(',')
        self.redis_input_stream_prefix = os.getenv('REDIS_INPUT_STREAM_PREFIX', 'objecttracker')
        self.path_to_model_json = os.getenv('PATH_TO_MODEL_JSON')
        self.store_video = os.getenv('STORE_WHOLE_VIDEO', 'false').lower() == 'false'


    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = AnomalyConfig()
        return cls.instance