from movementpredictor.evaluation.eval_config import EvalConfig
from movementpredictor.anomalydataset.dataset_creation import get_detected_anomalies_from_label_box, create_datasniplets
import os


evalconfig = EvalConfig()


def main():
    ids_of_interest, labels, time_intervals = get_detected_anomalies_from_label_box(evalconfig.path_label_box, evalconfig.camera)
    path_store = os.path.join("movementpredictor/anomalydataset/traffic_anomaly_dataset", evalconfig.camera, "testdata")
    create_datasniplets(ids_of_interest, labels, time_intervals, evalconfig.path_sae_dump, path_store)


if __name__ == "__main__":
    main()