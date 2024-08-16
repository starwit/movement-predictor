import os
from typing import Dict, Optional
from visionapi.common_pb2 import ModelInfo

import tomlkit

class ModelInfoParser:
    def __init__(self):
        self.pkg_meta: Dict[str, str] = self._get_project_meta()

    def _get_project_meta(self, pyproj_path: str = "./pyproject.toml") -> Dict[str, str]:
        if os.path.exists(pyproj_path):
            with open(pyproj_path, "r") as pyproject:
                file_contents = pyproject.read()
            return tomlkit.parse(file_contents)["tool"]["poetry"]
        else:
            return {}

    def parse(self):
        model_info = ModelInfo();
        model_info.name = self.pkg_meta.get("name")
        model_info.version = self.pkg_meta.get("version")
        return model_info