import logging
import json
from Training.RecurrentAE import AE
import torch
import os
import time
from pathlib import Path


log = logging.getLogger(__name__)


class CheckLoop():
    time_step_millisec = 20000


    def __init__(self, pipelineConnector, pathModelParameters):
        self.pipelineConnector = pipelineConnector
        parameters = self.read_json(pathModelParameters)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AE.LSTM_AE(parameters["dimension_latent_space"]).to(device)

        try:
            weights = torch.load(parameters["path_to_model"], map_location=torch.device(device))
        except:
            log.error(f"Model weights not found at {parameters["path_to_model"]}")
            exit(1)
        try:    
            self.model.load_state_dict(weights)
        except:
            log.error(f"Model weights do not fit model architecture LSTM_AE")
            exit(1)

        log.info("starting anomaly detection with parameters:")
        for key in parameters.keyset():
            log.info(key + ": " + str(parameters[key]))


    def read_json(self, file_path):
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
        

    def check_loop(self):
        