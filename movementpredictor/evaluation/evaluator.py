from pathlib import Path
from movementpredictor.evaluation.eval_config import EvalConfig
from typing import List, Tuple
import logging
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from movementpredictor.evaluation.main_eval_prep import make_combined_name


log = logging.getLogger(__name__)
evalconfig = EvalConfig()

groups = {
    0 : [-1],                 # tracking or detection mistake
    1 : [0],                 # false positive
    2 : [6, 7, 8, 9, 13, 16],                 # rather uninteresting anomaly
    3 : [1, 2, 3, 5, 10, 17, 19],                 # interesting anomaly
    4 : [4, 11, 12, 14, 15, 18]                  # dangerous behaviour
}


class PredictedTrajectory:

    def __init__(self, id: str, anomaly_measures: List[float]):
        self.obj_id = id
        self.measures = anomaly_measures
        self.label = PredictedTrajectory._add_label(id)
        self.anomaly_group = PredictedTrajectory.add_anomaly_group(self.label)

    @staticmethod
    def _add_label(id: str):
        path_label = os.path.join(evalconfig.path_label_box, evalconfig.camera, id, "labeldata.json")
        with open(path_label, "r") as json_file:
            labeldata = json.load(json_file)
        return labeldata["label"]

    @staticmethod
    def add_anomaly_group(label: int):
        for group, labels in groups.items():
            if label in labels:
                return group
        log.error("label " + str(label) + " has not been sorted into a group") 
        exit(1)


def all_predicted_ids_with_group(path_label_box, camera):
    ids = []
    class_labels = []
    path_label_box = os.path.join(path_label_box, camera)

    for entry in os.listdir(path_label_box):
        full_path = os.path.join(path_label_box, entry, "labeldata.json")
        with open(full_path, "r") as json_file:
            labeldata = json.load(json_file)
        label = PredictedTrajectory.add_anomaly_group(labeldata["label"])
        if label == 0:
            continue
        ids.append(labeldata["obj_id"])
        class_labels.append(label)
    
    return ids, class_labels


def all_anomaly_ids(all_ids, all_group_labels, good_class_start):
    anomaly_ids = []
    for car_id, group in zip(all_ids, all_group_labels):
        if group >= good_class_start:
            anomaly_ids.append(car_id)
    return anomaly_ids


good_class_start: int = 4
min_num_anomaly_frames: int = 4
all_ids, all_group_labels = all_predicted_ids_with_group(evalconfig.path_label_box, evalconfig.camera)
ground_truth = all_anomaly_ids(all_ids, all_group_labels, good_class_start)


def get_trajectories(path_predictions):
    trajectories: List[PredictedTrajectory] = []

    with open(path_predictions, 'r', encoding='utf-8') as f:
        try:
            anomaly_predictions = json.load(f)
            predictions = anomaly_predictions["predictions"]
            
        except json.JSONDecodeError as e:
            log.error(f"could not load json file {path_predictions}: {e}")
            exit(1)
        
        for id, trajectory in predictions.items():
            trajectories.append(PredictedTrajectory(id, trajectory["distances"] if "distances" in trajectory.keys() else trajectory["zscores"]))
    
    return trajectories


def calculate_PR_curve(trajectories: List[PredictedTrajectory]) -> List[Tuple[float, float]]:
    y_true = []
    scores = []
    traj_map = {t.obj_id: t for t in trajectories}

    for car_id in all_ids:
        y_true.append(1 if car_id in ground_truth else 0)
        if car_id not in traj_map or len(traj_map[car_id].measures) < min_num_anomaly_frames:
            scores.append(0.0)
        else:
            sorted_measures = sorted(traj_map[car_id].measures, reverse=True)
            scores.append(sorted_measures[min_num_anomaly_frames - 1])

    y_true = np.array(y_true)
    scores = np.clip(scores, 0, np.finfo(np.float32).max)
    scores = np.array(scores, dtype=np.float32)

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    pr_points = list(zip(recall, precision))

    return pr_points


def average_pr_curves(pr_curves, n_points=100):
    '''average by interpolation of the precision for equal recall scores'''
    recall_common = np.linspace(0.0, 1.0, n_points)
    
    all_precision = []
    for curve in pr_curves:
        recalls   = np.array([p for p,_ in curve])
        precisions= np.array([p for _,p in curve])
        
        order = np.argsort(recalls)
        recalls = recalls[order]
        precisions = precisions[order]
        
        p_interp = np.interp(
            recall_common,
            recalls,
            precisions,
            left=precisions[0],
            right=precisions[-1]
        )
        all_precision.append(p_interp)
    
    all_precision = np.stack(all_precision, axis=0)  # shape (n_curves, n_points)
    
    mean_precision = all_precision.mean(axis=0)
    std_precision  = all_precision.std(axis=0)
    
    return recall_common, mean_precision, std_precision


def mean_and_variance_PR_curve(trajectories_of_all_runs: List[List[PredictedTrajectory]], model_name: str, show: bool = False):
    pr_curves: List[np.array] = []

    for trajectory in trajectories_of_all_runs:
        pr_curves.append(np.array(calculate_PR_curve(trajectory)))
    
    recall, mean_p, std_p = average_pr_curves(pr_curves, n_points=100)
    
    plt.plot(recall, mean_p, label=model_name)
    plt.fill_between(recall, mean_p-std_p, mean_p+std_p, alpha=0.3)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Start class positives: " + str(good_class_start) + ", min_anomal_detections: " + str(min_num_anomaly_frames))
    plt.legend()
    plt.grid(True)
    
    if show:
        path = os.path.join("movementpredictor/evaluation/plots", evalconfig.camera)
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, "pr-curves-" + str(good_class_start) + "-" + str(min_num_anomaly_frames) + ".png"))
        plt.show()


def find_matching_files(search_dir: str, prefix: str) -> List[str]:
    p = Path(search_dir)
    return [f for f in p.iterdir() if f.is_file() and f.name.startswith(prefix)]


model_name = "1sec_MobileNet_v3_symmetric_prob"
path_list_predictions = find_matching_files(evalconfig.path_store_anomalies, model_name)

trajectories_of_all_runs = []
for path in path_list_predictions:
    trajectories_of_all_runs.append(get_trajectories(path))

mean_and_variance_PR_curve(trajectories_of_all_runs, model_name)


model_name = "1sec_MobileNet_v3_asymmetric_prob"
path_list_predictions = find_matching_files(evalconfig.path_store_anomalies, model_name)

trajectories_of_all_runs = []
for path in path_list_predictions:
    trajectories_of_all_runs.append(get_trajectories(path))

mean_and_variance_PR_curve(trajectories_of_all_runs, model_name)


model_name = "2sec_MobileNet_v3_symmetric_prob"
path_list_predictions = find_matching_files(evalconfig.path_store_anomalies, model_name)

trajectories_of_all_runs = []
for path in path_list_predictions:
    trajectories_of_all_runs.append(get_trajectories(path))

mean_and_variance_PR_curve(trajectories_of_all_runs, model_name)


model_name = "2sec_MobileNet_v3_asymmetric_prob"
path_list_predictions = find_matching_files(evalconfig.path_store_anomalies, model_name)

trajectories_of_all_runs = []
for path in path_list_predictions:
    trajectories_of_all_runs.append(get_trajectories(path))

mean_and_variance_PR_curve(trajectories_of_all_runs, model_name)


model_name = "nearest_neighbor_analysis"
path_list_predictions = find_matching_files(evalconfig.path_store_anomalies, model_name)

trajectories_of_all_runs = []
for path in path_list_predictions:
    trajectories_of_all_runs.append(get_trajectories(path))

mean_and_variance_PR_curve(trajectories_of_all_runs, model_name, show=True)