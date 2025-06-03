from movementpredictor.config import ModelConfig
from movementpredictor.anomalydataset.dataset_creation import create_train_data
import os


config = ModelConfig()


def main():
    path_store = os.path.join("movementpredictor/anomalydataset/traffic_anomaly_dataset", config.camera, "traindata")
    create_train_data(config.path_sae_data_train, path_store)


if __name__ == "__main__":
    main()