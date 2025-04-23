from movementpredictor.cnn import training
from movementpredictor.config import ModelConfig

import logging

log = logging.getLogger(__name__)
config = ModelConfig()


def main():
    
    model, history = training.trainAndStoreCNN(config.path_store_data, config.path_model, config.model_architecture, config.output_distribution, config.pixel_per_axis) 
    training.store_parameters(history, config)

    training.plot_loss_curve(history, config.path_plots)
    model.eval()
    
if __name__ == "__main__":
    main()