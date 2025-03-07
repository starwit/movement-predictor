import sys
sys.path.append('/home/starwit01/workspaces/hanna/movement-predictor')

from movementpredictor.cnn import probabilistic_regression
from movementpredictor.config import ModelConfig
from movementpredictor.data import dataset

import logging

log = logging.getLogger(__name__)
config = ModelConfig()


def main():

    model, history = probabilistic_regression.trainAndStoreCNN(config.path_store_data, config.path_model)
    probabilistic_regression.store_parameters(history, config)
    model.eval()
    probabilistic_regression.plot_loss_curve(history)
    
if __name__ == "__main__":
    main()