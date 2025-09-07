from collections import defaultdict
from pathlib import Path
from movementpredictor.evaluation.eval_config import EvalConfig
from typing import List, Tuple
import logging
import os
import json
import numpy as np


log = logging.getLogger(__name__)
evalconfig = EvalConfig()

mapping_path = Path("movementpredictor/anomalydataset/traffic_anomaly_dataset/relevance_mapping.json")
with mapping_path.open("r", encoding="utf-8") as f:
    raw = json.load(f)

# convert keys to ints and build groups dict
groups = {int(k): v for k, v in raw.items()}

# add the “mistake” group for -1
groups[-1] = [-1]


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
        label = labeldata["label"]
        return label if label != "None" else -1

    @staticmethod
    def add_anomaly_group(label: int):
        if label == "None": label = -1
        for group, labels in groups.items():
            if label in labels:
                return group
        log.error("label " + str(label) + " has not been sorted into a group") 
        exit(1)


def all_predicted_ids_with_group(path_label_box, camera):
    ids = []
    group_labels = []
    event_labels = []
    path_label_box = os.path.join(path_label_box, camera)

    for entry in os.listdir(path_label_box):
        full_path = os.path.join(path_label_box, entry, "labeldata.json")

        with open(full_path, "r") as json_file:
            labeldata = json.load(json_file)
         
        event_labels.append(labeldata["label"] if labeldata["label"] != "None" else -1)
        label = PredictedTrajectory.add_anomaly_group(labeldata["label"])
        ids.append(labeldata["obj_id"])
        group_labels.append(label)
    
    return ids, group_labels, event_labels


def calculate_threshold(trajectories: List[PredictedTrajectory], num_anomalous_trajectories, min_num_anomal_points):
    dist_obj_pairs = [(measure, trajectory.obj_id) for trajectory in trajectories for measure in  trajectory.measures]
    dist_obj_pairs.sort(reverse=True, key=lambda x: x[0])

    obj_id_with_dists = defaultdict(list)
    anomaly_obj_ids = set()
    threshold_dists = None

    for dist, obj_id in dist_obj_pairs:
        obj_id_with_dists[obj_id].append(dist)
        if len(obj_id_with_dists[obj_id]) >= min_num_anomal_points:
            anomaly_obj_ids.add(obj_id)
        if len(anomaly_obj_ids) >= num_anomalous_trajectories:
            threshold_dists = dist
            break

    log.info("Distance-threshold: " + str(threshold_dists))
    return threshold_dists, anomaly_obj_ids


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


def get_y_true(all_group_labels: List[int], include_mistakes: bool = False) -> List[int]:
    if include_mistakes:
        return all_group_labels
    else:
        return [group_label for group_label in all_group_labels if group_label != -1]


def get_rels(all_group_labels: List[int], include_mistakes: bool = False) -> List[int]:
    return [max(0, anomaly_group) for anomaly_group in get_y_true(all_group_labels, include_mistakes)]    


def score_trajectory(trajectory: PredictedTrajectory, scoring: str = "weighted-avg", exp_para=None):
    sorted_measures = sorted(trajectory.measures, reverse=True)

    if scoring == "avg":
        score = np.mean(np.array(sorted_measures))

    elif scoring == "weighted-avg":
        sorted_measures = np.array(sorted_measures)
        score = 0
        sum_weights = 0
        length_trajectory = len(sorted_measures)
        for rank, measure in enumerate(sorted_measures):
            weight = (length_trajectory-rank+1)/length_trajectory
            score += (measure*weight)
            sum_weights += weight
        score = score/sum_weights

    elif scoring == "exp-weighted-avg":
        sorted_measures = np.array(sorted_measures)
        score = 0
        sum_weights = 0
        for rank, measure in enumerate(sorted_measures):
            weight = (0.5 if exp_para is None else exp_para)**rank
            score += (measure*weight)
            sum_weights += weight
        score = score/sum_weights

    else:
        log.error("scoring has to be avg, weighted-avg or exp-weighted-avg")
        exit(1)
    
    return score


def get_scores_full_trajectory(trajectories: List[PredictedTrajectory], all_ids: List[str], all_group_labels: List[int], scoring: str = "weighted-avg", 
                               exp_para=None, include_mistakes: bool = False):
    scores = []
    traj_map = {t.obj_id: t for t in trajectories}

    for car_id, group_label in zip(all_ids, all_group_labels):
        if group_label == -1 and not include_mistakes:  # sort out tracking and detection mistakes
            continue

        if car_id not in traj_map.keys():
            log.debug("no predictions for " + car_id)
            scores.append(0)

        else:
            score = score_trajectory(traj_map[car_id], scoring, exp_para)
            scores.append(score)
    
    rng = np.random.default_rng(seed=42)  
    noise = rng.normal(loc=0.0, scale=1e-9, size=len(scores))
    scores = np.array(scores, dtype=np.float64)
    scores[scores != 0] += noise[scores != 0]

    return scores


def get_scores(trajectories: List[PredictedTrajectory], all_ids: List[str], all_group_labels: List[int], min_num_anomaly_frames: int = None, 
                     portion: float = None, scoring: str = "avg", include_mistakes: bool = False):
    ''' scoring: calculation method for the score - 'avg', 'min', 'weighted-avg' or 'med' '''
    scores = []
    traj_map = {t.obj_id: t for t in trajectories}

    if portion is None and min_num_anomaly_frames is None:
        log.error("either portion or min_num_anomaly_frames has to be set")
        exit(1)

    for car_id, group_label in zip(all_ids, all_group_labels):
        if group_label == -1 and not include_mistakes:  # sort out tracking and detection mistakes
            continue

        if car_id not in traj_map.keys():
            log.debug("no predictions for " + car_id)
            scores.append(0)

        else:
            sorted_measures = sorted(traj_map[car_id].measures, reverse=True)
            cut_off = min_num_anomaly_frames if min_num_anomaly_frames is not None else max(1, round(portion*len(sorted_measures)))

            if scoring == "min-exc":
                if len(traj_map[car_id].measures) < cut_off:
                    log.debug("not enough predictions for " + car_id + ": " + str(traj_map[car_id].measures))
                    score = 0
                else:
                    score = sorted_measures[cut_off - 1]
            
            elif scoring == "min-inc":
                score = sorted_measures[min(cut_off, len(sorted_measures)) - 1]

            elif scoring == "avg":
                score = np.mean(np.array(sorted_measures[:min(cut_off, len(sorted_measures))]))
            
            elif scoring == "med":
                score = np.median(np.array(sorted_measures[:min(cut_off, len(sorted_measures))]))
                
            elif scoring == "weighted-avg":
                sorted_measures = np.array(sorted_measures[:min(cut_off, len(sorted_measures))])
                score = 0
                sum_weights = 0
                length_trajectory = len(sorted_measures)
                for rank, measure in enumerate(sorted_measures):
                    weight = (length_trajectory-rank+1)/length_trajectory
                    score += (measure*weight)
                    sum_weights += weight
                score = score/sum_weights

            elif scoring == "exp-weighted-avg":
                sorted_measures = np.array(sorted_measures[:min(cut_off, len(sorted_measures))])
                score = 0
                sum_weights = 0
                for rank, measure in enumerate(sorted_measures):
                    weight = (0.95)**rank
                    score += (measure*weight)
                    sum_weights += weight
                score = score/sum_weights

            else:
                log.error("scoring has to be min, avg, weighted-sum or med")
                exit(1)

            scores.append(score)


    rng = np.random.default_rng(seed=42)  
    noise = rng.normal(loc=0.0, scale=1e-9, size=len(scores))
    scores = np.array(scores, dtype=np.float64)
    scores[scores != 0] += noise[scores != 0]

    log.debug(str(len([score for score in scores if score == 0])) + " trajectories could not be detected because there are not enough predictions")
    return  scores


def find_matching_files(search_dir: str, prefix: str) -> List[str]:
    p = Path(search_dir)
    return [f for f in p.iterdir() if f.is_file() and f.name.startswith(prefix) and f.name.endswith("all_labeled_data.json")]


def find_matching_files_top_k(search_dir: str, prefix: str, num_anomaly_points: int) -> List[str]:
    p = Path(search_dir)
    return [f for f in p.iterdir() if f.is_file() and f.name.startswith(prefix) and f.name.endswith(str(num_anomaly_points) + ".json")]


