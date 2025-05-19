from typing import List
from movementpredictor.cnn import model_architectures
from movementpredictor.cnn.inferencing import inference_with_stats
from movementpredictor.data import dataset
from movementpredictor.evaluation.eval_config import EvalConfig
from movementpredictor.evaluation.eval_prep import store_predictions

import os

evalconfig = EvalConfig()


def get_obj_ids_from_label_box(path_label_box: str, camera: str) -> List[str]:
    path_label_box = os.path.join(path_label_box, camera)
    ids = os.listdir(path_label_box)
    return ids


def main():
    ids_of_interest = get_obj_ids_from_label_box(evalconfig.path_label_box, evalconfig.camera)
    model = model_architectures.get_model(evalconfig.model_architecture, evalconfig.output_distribution, path_model=evalconfig.path_model)
    model.eval()

    ds = dataset.getTorchDataSet(evalconfig.path_test_data, ids_of_interest=ids_of_interest)
    test = dataset.getTorchDataLoader(ds, shuffle=False)

    samples_with_stats = inference_with_stats(model, test)
    print("total test samples: " + str(len(samples_with_stats)))

    store_predictions(samples_with_stats, evalconfig.path_store_anomalies, evalconfig.path_model, evalconfig.num_anomalies)


if __name__ == "__main__":
    main()