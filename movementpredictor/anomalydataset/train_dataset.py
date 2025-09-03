import os

from movementpredictor.config import ModelConfig
from movementpredictor.anomalydataset import dataset_creation


config = ModelConfig()


def main():
    path_store = os.path.join("movementpredictor/anomalydataset/traffic_anomaly_dataset", config.camera, "traindata")
    dataset_creation.create_raw_dataset(
        paths_sae_dumps=[config.path_sae_data_train],
        path_store=path_store,
        max_frames=None,                 
        write_mode="shards",              # "files" | "shards" | "both"
        shard_prefix="frames",
        max_shard_size_bytes=1_000_000_000,
        max_samples_per_shard=10000,
    )


if __name__ == "__main__":
    main()

