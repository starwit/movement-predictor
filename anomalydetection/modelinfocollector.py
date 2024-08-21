import os
import logging
import json
from typing import Dict
import tomlkit
from pathlib import Path
from visionapi.common_pb2 import ModelInfo
from anomalydetection.config import AnomalyDetectionConfig

log = logging.getLogger(__name__)

class ModelInfoCollector:

    def __init__(self, CONFIG: AnomalyDetectionConfig) -> None:
        self.pkg_meta: Dict[str, str] = self._read_project_meta()
        self.model_parameters = self._read_model_info_from_json(CONFIG.path_to_model_config)
        self.model_info: ModelInfo = self._get_model_info()

    def _get_model_info(self) -> ModelInfo:
        model_info = ModelInfo();
        model_info.name = self.model_parameters["model_name"]
        #TODO get model version
        model_info.version = self.pkg_meta.get("version", "")
        return model_info 

    def _read_project_meta(self, pyproj_path: str = "./pyproject.toml") -> Dict[str, str]:
        if os.path.exists(pyproj_path):
            with open(pyproj_path, "r") as pyproject:
                file_contents = pyproject.read()
            return tomlkit.parse(file_contents)["tool"]["poetry"]
        else:
            return {}
    
    def _read_model_info_from_json(self, file_path):
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                log.error(f"Could not find model parameters at {file_path}")
                return None
            with file_path.open('r') as file:
                return json.load(file)
        except IOError as e:
            log.error(f"Could not read model parameters {file_path}")
            log.debug(e)
            return None
        