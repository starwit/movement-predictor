from movementpredictor.data.data_prep import build_train_dataset, build_test_dataset
from movementpredictor.cnn.start_training import train_movement_predictor
from movementpredictor.anomalydetection.run_inference import run_inference_and_calc_anomaly_threshold
from movementpredictor.config import ModelConfig

import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

config = ModelConfig()


def main():

    if "prepare" in config.steps:
        build_train_dataset()
        build_test_dataset()

    if "train" in config.steps:
        train_movement_predictor()

    if "threshold" in config.steps:
        run_inference_and_calc_anomaly_threshold()

if __name__ == "__main__":
    main()