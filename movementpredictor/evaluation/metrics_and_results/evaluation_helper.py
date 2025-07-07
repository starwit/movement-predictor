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

groups = {      
    0 : [-1],                 # tracking or detection mistake
    1 : [0, 1],                 # false positive or uninteresting 
    2 : [2, 3, 4, 6, 8, 15, 17, 27],               # rather uninteresting anomaly
    3 : [5, 7, 10, 12, 16, 24, 26, 30],                # interesting anomaly
    4 : [9, 11, 14, 18, 21, 29, 31],                           # highly interesting anomaly
    5 : [13, 19, 20, 22, 23, 25, 28, 32]                 # dangerous behaviour
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


def get_y_true(all_group_labels: List[int]):
    return [group_label for group_label in all_group_labels if group_label != 0]


#def get_rels(all_group_labels: List[int], exp_relevance: bool = True):
 #   return [2**max(0, anomaly_group - 1) -1 if exp_relevance else max(0, anomaly_group - 1) for anomaly_group in get_y_true(all_group_labels)]

def get_rels(all_group_labels: List[int]):
    return [max(0, anomaly_group - 1) for anomaly_group in get_y_true(all_group_labels)]


def get_scores_num_trajectories_based(trajectories: List[PredictedTrajectory], num_anomaly_trajectories: int, min_num_anomaly_frames: int, all_ids: List[str], all_group_labels: List[int],
               weight_length: float = 0.5):
    scores = []
    threshold, top_k_ids = calculate_threshold(trajectories, num_anomaly_trajectories, min_num_anomaly_frames)
    detections: List[PredictedTrajectory] = []

    for trajectory in trajectories:
        high_measures = [measure for measure in trajectory.measures if measure >= threshold]
        if len(high_measures) >= min_num_anomaly_frames:
            detections.append(PredictedTrajectory(trajectory.obj_id, high_measures))

    max_anomal_trajectory_len = max([len(trajectory.measures) for trajectory in detections])
    max_measure = max([max(trajectory.measures) for trajectory in detections])
    traj_map = {t.obj_id: t for t in detections}

    for car_id, group_label in zip(all_ids, all_group_labels):
        if group_label == 0:  # sort out tracking and detection mistakes
            continue

        if car_id not in traj_map.keys():
            log.debug("no predictions for " + car_id)
            scores.append(0)

        else:
            score = weight_length*(len(traj_map[car_id].measures)/max_anomal_trajectory_len) + (1-weight_length)*(max(traj_map[car_id].measures)/max_measure)
            scores.append(score)
    
    return np.array(scores)


def score_trajectory(trajectory: PredictedTrajectory, scoring: str = "weighted-avg", exp_para=None):
    sorted_measures = sorted(trajectory.measures, reverse=True)
    #sorted_measures = sorted_measures[:min(len(sorted_measures), 50)]

    if scoring == "avg":
        score = np.mean(np.array(sorted_measures))

    elif scoring == "weighted-avg":
        sorted_measures = np.array(sorted_measures)
        score = 0
        sum_weights = 0
        length_trajectory = len(sorted_measures)
        for rank, measure in enumerate(sorted_measures):
            weight = (length_trajectory-rank)/length_trajectory
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



def get_scores_full_trajectory(trajectories: List[PredictedTrajectory], all_ids: List[str], all_group_labels: List[int], scoring: str = "weighted-avg", exp_para=None):
    scores = []
    traj_map = {t.obj_id: t for t in trajectories}

    for car_id, group_label in zip(all_ids, all_group_labels):
        if group_label == 0:  # sort out tracking and detection mistakes
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



def get_scores_top_k(trajectories: List[PredictedTrajectory], min_num_anomaly_frames: int, all_ids: List[str], all_group_labels: List[int],
               scoring: str = "avg", remove_undetected = False):
    ''' scoring: calculation method for the score - 'avg', 'min', 'weighted-avg' or 'med' '''
    scores = []
    traj_map = {t.obj_id: t for t in trajectories}

    for car_id, group_label in zip(all_ids, all_group_labels):
        if group_label == 0:  # sort out tracking and detection mistakes
            continue

        if car_id not in traj_map.keys():
            log.debug("no predictions for " + car_id)
            if remove_undetected:
                continue
            scores.append(0)

        #elif len(traj_map[car_id].measures) < min_num_anomaly_frames:
         #   log.debug("not enough predictions for " + car_id + ": " + str(traj_map[car_id].measures))
          #  if remove_undetected:
           #     continue
            #scores.append(0)

        else:
            sorted_measures = sorted(traj_map[car_id].measures, reverse=True)
            cut_off = min_num_anomaly_frames if min_num_anomaly_frames >= 1 else max(1, round(min_num_anomaly_frames*len(sorted_measures)))

            if scoring == "min-exc":
                if len(traj_map[car_id].measures) < cut_off:
                    log.debug("not enough predictions for " + car_id + ": " + str(traj_map[car_id].measures))
                    if remove_undetected:
                        continue
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
                    weight = (length_trajectory-rank)/length_trajectory
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


def get_scores_percentile(trajectories: List[PredictedTrajectory], percentile: int, all_ids: List[str], all_group_labels: List[int],
               scoring: str = "avg"):
    ''' scoring: calculation method for the score - 'avg', 'min', 'weighted-sum' or 'med' '''
    scores = []
    traj_map = {t.obj_id: t for t in trajectories}


    for car_id, group_label in zip(all_ids, all_group_labels):
        if group_label == 0:  # sort out tracking and detection mistakes
            continue

        if car_id not in traj_map.keys():
            log.debug("no predictions for " + car_id)
            scores.append(0)

        else:
            num_anomaly_frames = int(len(traj_map[car_id].measures)*(100-percentile)/100)
            if num_anomaly_frames == 0:
                scores.append(0)
                continue
            sorted_measures = sorted(traj_map[car_id].measures, reverse=True)

            if scoring == "min":
                score = sorted_measures[num_anomaly_frames - 1]
            elif scoring == "avg":
                score = np.mean(np.array(sorted_measures[:num_anomaly_frames]))
            elif scoring == "med":
                score = np.median(np.array(sorted_measures[:num_anomaly_frames]))
            elif scoring == "weighted-sum":
                sorted_measures = np.array(sorted_measures[:num_anomaly_frames])
                score = 0
                for rank, measure in enumerate(sorted_measures):
                    score += measure*(num_anomaly_frames-rank)/num_anomaly_frames
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


