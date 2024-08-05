from typing import ClassVar
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Annotated
from visionlib.pipeline.settings import LogLevel, YamlConfigSettingsSource
import os


class RedisConfig(BaseModel):
    host: str = 'localhost'
    port: Annotated[int, Field(ge=1, le=65536)] = 6379
    stream_id: str = 'stream1'
    input_stream_prefix: str = 'objecttracker'
    output_stream_prefix: str = 'anomalydetection'

class AnomalyDetectionConfig(BaseSettings):
    log_level: LogLevel = LogLevel.WARNING
    redis: RedisConfig = RedisConfig()
    prometheus_port: Annotated[int, Field(ge=1024, le=65536)] = 8000
    model_config = SettingsConfigDict(env_nested_delimiter='__')
    path_to_model_config: ClassVar[str] = "/home/hanna/workspaces/AETrajectories/AEsAnomalyDetection/RecurrentAE/parameters.json" #= os.getenv("PATH_TO_MODEL_CONFIG")
    whole_video: bool = False

    
    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls), file_secret_settings)