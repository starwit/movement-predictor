from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Annotated
from visionlib.pipeline.settings import LogLevel, YamlConfigSettingsSource


class RedisConfig(BaseModel):
    host: str = 'localhost'
    port: Annotated[int, Field(ge=1, le=65536)] = 6379
    stream_id: str = 'stream1'
    stream_prefix: str = 'undefined'

class ModelConfig(BaseModel):
    anomaly_threshold_test: float = -1
    parameters_path: Path = Path("model/parameters.json")
    weights_path: Path = Path("model/model_weights.pth")
    background_path: Path = Path("model/frame.pth")

class AnomalyDetectionConfig(BaseSettings):
    log_level: LogLevel = LogLevel.INFO
    redisIn: RedisConfig = RedisConfig()
    redisOut: RedisConfig = RedisConfig()
    prometheus_port: Annotated[int, Field(ge=1024, le=65536)] = 8000
    model: ModelConfig = ModelConfig()
    filtering: bool = True

    # `model_config` refers to the pydantic model and has nothing to do with `model` above
    model_config = SettingsConfigDict(env_nested_delimiter='__')
    
    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls), file_secret_settings)