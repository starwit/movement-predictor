from typing import List
import json
import pandas as pd
from movementpredictor.evaluation.eval_config import EvalConfig
from movementpredictor.anomalydataset import dataset_creation
import os


evalconfig = EvalConfig()


def get_detected_anomalies_from_label_box(path_label_box: str, camera: str) -> List[str]:
    path_label_box = os.path.join(path_label_box, camera)
    ids = os.listdir(path_label_box)
    labels = []
    time_intervals = []
    for obj_id in ids:
        with open(os.path.join(path_label_box, obj_id, "labeldata.json"), "r", encoding="utf-8") as f:
            label_data = json.load(f) 
        labels.append(label_data["label"])
        time_intervals.append(label_data["time_interval"])
    return ids, labels, time_intervals


def store_anomaly_annotations(path_store: str):
    '''store anomaly predictions'''
    ids_of_interest, labels, time_intervals = get_detected_anomalies_from_label_box(evalconfig.path_label_box, evalconfig.camera)
    
    anomaly_intervals = []
    for obj_id, label, interval in zip(ids_of_interest, labels, time_intervals):
        if label != 0:
            anomaly_intervals.append({
                "object_id": obj_id,
                "start_timestamp": interval[0],
                "end_timestamp": interval[1],
                "label": label
            })

    df = pd.DataFrame(anomaly_intervals)
    df.to_csv(os.path.join(path_store, "anomaly-labels.csv"), index=False)


def main():
    path_store = os.path.join("movementpredictor/anomalydataset/traffic_anomaly_dataset", evalconfig.camera, "testdata")

    dataset_creation.create_raw_dataset(
        paths_sae_dumps=evalconfig.path_sae_dumps,
        path_store=path_store,
        max_frames=5000,                 # or None
        write_mode="shards",              # "files" | "shards" | "both"
        shard_prefix="frames",
        max_shard_size_bytes=1_000_000_000,
        max_samples_per_shard=10000,
    )

    store_anomaly_annotations(
        path_store=path_store
    )



if __name__ == "__main__":
    main()