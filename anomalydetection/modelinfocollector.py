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
    __instance = None
  
    @staticmethod
    def get_model_parameter():
        return ModelInfoCollector._getInstance().model_parameters

    @staticmethod
    def get_model_info() -> ModelInfo:
        model_info = ModelInfo();
        model_info.name = ModelInfoCollector._getInstance().model_parameters["model_name"]
        #TODO get model version
        model_info.version = ModelInfoCollector._getInstance().pkg_meta.get("version")
        return model_info 
    
    @staticmethod
    def _getInstance():
        if ModelInfoCollector.__instance == None:
            ModelInfoCollector()
        return ModelInfoCollector.__instance

    def __init__(self):
        if ModelInfoCollector.__instance != None:
            raise Exception("Singleton object already created!")
        else:     
            self.config = AnomalyDetectionConfig();
            self.pkg_meta: Dict[str, str] = self._read_project_meta()
            self.model_parameters = self._read_model_info_from_json(self.config.path_to_model_config)
            ModelInfoCollector.__instance = self

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
        