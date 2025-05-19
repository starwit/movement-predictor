from collections import Counter
from pathlib import Path
from movementpredictor.evaluation.eval_config import EvalConfig
from typing import List, Tuple
import logging
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score, dcg_score
from sklearn.metrics import precision_recall_curve, auc
from scipy.stats import rankdata


log = logging.getLogger(__name__)
evalconfig = EvalConfig()

groups = {      
    0 : [-1],                 # tracking or detection mistake
    1 : [0, 1, 3, 4],                 # false positive or uninteresting 
    2 : [2, 6, 15, 17, 27, 31],               # rather uninteresting anomaly
    3 : [5, 7, 8, 9, 10, 11, 12, 14, 16, 18, 26, 29, 30],                # interesting anomaly
    4 : [13, 19, 20, 21, 22, 23, 24, 25, 28, 32]                 # dangerous behaviour
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
    group_labels = []
    event_labels = []
    path_label_box = os.path.join(path_label_box, camera)

    for entry in os.listdir(path_label_box):
        full_path = os.path.join(path_label_box, entry, "labeldata.json")

        with open(full_path, "r") as json_file:
            labeldata = json.load(json_file)

        event_labels.append(labeldata["label"])
        label = PredictedTrajectory.add_anomaly_group(labeldata["label"])
        ids.append(labeldata["obj_id"])
        group_labels.append(label)
    
    return ids, group_labels, event_labels


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


def get_rels(all_group_labels: List[int], squared_relevance: bool):
    return [(anomaly_group - 1) ** 2 if squared_relevance else (anomaly_group - 1) for anomaly_group in get_y_true(all_group_labels)]


def get_scores(trajectories: List[PredictedTrajectory], min_num_anomaly_frames: int, all_ids: List[str], all_group_labels: List[int],
               scoring: str = "min", remove_undetected = False):
    ''' scoring: calculation method for the score - 'avg', 'min' or 'med' '''
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

        elif len(traj_map[car_id].measures) < min_num_anomaly_frames:
            log.debug("not enough predictions for " + car_id + ": " + str(traj_map[car_id].measures))
            if remove_undetected:
                continue
            scores.append(0)

        else:
            sorted_measures = sorted(traj_map[car_id].measures, reverse=True)
            if scoring == "min":
                score = sorted_measures[min_num_anomaly_frames - 1]
            elif scoring == "avg":
                score = np.mean(np.array(sorted_measures[:min_num_anomaly_frames]))
            elif scoring == "med":
                score = np.median(np.array(sorted_measures[:min_num_anomaly_frames]))
            else:
                log.error("scoring has to be min, avg or med")
                exit(1)

            scores.append(score)


    rng = np.random.default_rng(seed=42)  
    noise = rng.normal(loc=0.0, scale=1e-9, size=len(scores))
    scores = np.array(scores, dtype=np.float64)
    scores[scores != 0] += noise[scores != 0]

    log.debug(str(len([score for score in scores if score == 0])) + " trajectories could not be detected because there are not enough predictions")
    return  scores


def calculate_PR_curve(trajectories: List[PredictedTrajectory], good_class_start: int, min_num_anomaly_frames: int, 
                       all_ids: List[str], all_group_labels: List[int], scoring: str) -> List[Tuple[float, float]]:
    
    y_true = get_y_true(all_group_labels)
    scores = get_scores(trajectories, min_num_anomaly_frames, all_ids, all_group_labels, scoring)
    y_true = np.array([class_label >= good_class_start for class_label in y_true])

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)

    return precision, recall, thresholds, pr_auc


def best_F1_scores_plot(trajectories_of_all_runs: List[List[PredictedTrajectory]], model_name: str, good_class_start: int,
            all_ids: List[str], all_group_labels: List[int], scoring: str, show: bool = False):
    best_f1 = []
    best_f1_std = []
    best_f1_threshold = []
    best_f1_threshold_std = []

    max_min_anomaly_frames = 80

    for i in range(max_min_anomaly_frames):
        best_f1_temp = []
        best_f1_threshold_temp = []

        for trajectory in trajectories_of_all_runs:
            precision, recall, thresholds, _ = calculate_PR_curve(trajectory, good_class_start, i+1, all_ids, all_group_labels, scoring)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

            best_index = np.argmax(f1_scores)
            best_f1_temp.append(f1_scores[best_index])
            best_f1_threshold_temp.append(thresholds[best_index])
        
        best_f1.append(np.mean(best_f1_temp))
        best_f1_threshold.append(np.mean(best_f1_threshold_temp))
        best_f1_std.append(np.std(best_f1_temp))
        best_f1_threshold_std.append(np.std(best_f1_threshold_temp))

    f1 = np.array(best_f1)
    f1_std = np.array(best_f1_std)
    plt.plot(range(1, max_min_anomaly_frames+1), f1, label=model_name)
    plt.fill_between(range(1, max_min_anomaly_frames+1), f1-0.5*f1_std, f1+0.5*f1_std, alpha=0.3)
    
    plt.xlabel('min num anomaly frames')
    plt.ylabel('best F1')
    plt.title("Best F1 - start class positives: " + str(good_class_start))
    plt.legend()
    plt.grid(True)
    
    if show:
        path = os.path.join("movementpredictor/evaluation/plots", evalconfig.camera)
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, "best-f1-" + str(good_class_start) + "-" + scoring + ".png"))
        plt.show()
        plt.close()
    
    index_best_of_all = np.argmax(f1)
    return f1[index_best_of_all], f1_std[index_best_of_all], best_f1_threshold[index_best_of_all], best_f1_threshold_std[index_best_of_all], index_best_of_all+1



def PR_AUCs(trajectories_of_all_runs: List[List[PredictedTrajectory]], model_name: str, good_class_start: int,
            all_ids: List[str], all_group_labels: List[int], scoring: str, show: bool = False):
    aucs = []
    aucs_std = []
    max_min_anomaly_frames = 80

    for i in range(max_min_anomaly_frames):
        aucs_temp = []
        for trajectory in trajectories_of_all_runs:
            _, _, _, pr_auc = calculate_PR_curve(trajectory, good_class_start, i+1, all_ids, all_group_labels, scoring)
            aucs_temp.append(pr_auc)

        aucs.append(np.mean(aucs_temp))
        aucs_std.append(np.std(aucs_temp))
    
    aucs = np.array(aucs)
    aucs_std = np.array(aucs_std)
    plt.plot(range(1, max_min_anomaly_frames+1), aucs, label=model_name)
    plt.fill_between(range(1, max_min_anomaly_frames+1), aucs-0.5*aucs_std, aucs+0.5*aucs_std, alpha=0.3)
    
    plt.xlabel('min num anomaly frames')
    plt.ylabel('PR-AUC')
    plt.title("PR-AUC - start class positives: " + str(good_class_start))
    plt.legend()
    plt.grid(True)
    
    if show:
        path = os.path.join("movementpredictor/evaluation/plots", evalconfig.camera)
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, "auc-pr-curve-" + str(good_class_start) + "-" + scoring + ".png"))
        plt.show()
        plt.close()


def mean_and_variance_PR_curve(trajectories_of_all_runs: List[List[PredictedTrajectory]], model_name: str, good_class_start: int,
                               min_num_anomaly_frames: int, all_ids: List[str], all_group_labels: List[int], scoring: str, show: bool = False):
    pr_curves: List[np.array] = []

    for trajectory in trajectories_of_all_runs:
        precision, recall, _, _ = calculate_PR_curve(trajectory, good_class_start, min_num_anomaly_frames, all_ids, all_group_labels, scoring)
        curve = list(zip(recall, precision))
        pr_curves.append(np.array(curve))
    
    #recall, mean_p, std_p = average_pr_curves(pr_curves, n_points=100)
    pr_curves_stacked = np.stack(pr_curves, axis=0)
    mean_curve = np.mean(pr_curves_stacked, axis=0)    # (n_points, 2)
    std_curve  = np.std(pr_curves_stacked, axis=0)

    precision_mean = mean_curve[:, 1]
    precision_std  = std_curve[:, 1]
    recall_mean    = mean_curve[:, 0]
    
    plt.plot(recall_mean, precision_mean, label=model_name)
    plt.fill_between(recall_mean, precision_mean-0.5*precision_std, precision_mean+0.5*precision_std, alpha=0.3)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0, 1.03)
    plt.title("Start class positives: " + str(good_class_start) + ", min_anomal_detections: " + str(min_num_anomaly_frames))
    plt.legend()
    plt.grid(True)
    
    if show:
        path = os.path.join("movementpredictor/evaluation/plots", evalconfig.camera)
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, "pr-curves-" + str(good_class_start) + "-" + str(min_num_anomaly_frames) + ".png"))
        plt.show()
        plt.close()


def dcg_curve(trajectories_of_all_runs: List[List[PredictedTrajectory]], model_name: str, all_ids: List[str], 
              all_group_labels: List[int], scoring: str, normalized: bool = False, show: bool = False, squared_relevance = True):
    curve_mean = []
    curve_std = []

    rels = get_rels(all_group_labels, squared_relevance)
    if normalized:
        ideal_order = np.sort(rels)[::-1]
        idcg = 0
        for i, rel in enumerate(ideal_order):
            idcg += rel/np.log2(i+2)
    else:
        idcg = None

    max_min_length = 80
    max_dcg, max_dcg_std, min_length = 0, 0, 0

    for i in range(max_min_length):
        dcg_mean, dcg_std = dcg_mean_and_std(trajectories_of_all_runs, i+1, all_ids, all_group_labels, scoring, rels, idcg)
        curve_mean.append(dcg_mean)
        curve_std.append(dcg_std)

        if dcg_mean > max_dcg:
            max_dcg = dcg_mean
            max_dcg_std = dcg_std
            min_length = i
    
    curve_mean = np.array(curve_mean)
    curve_std = np.array(curve_std)
    plt.plot(range(1, max_min_length+1), curve_mean, label=model_name)
    plt.fill_between(range(1, max_min_length+1), curve_mean-0.5*curve_std, curve_mean+0.5*curve_std, alpha=0.3)
    
    plt.xlabel('min length of anomal trajectory')
    plt.ylabel('NDCG' if normalized else 'DCG')
    plt.title("NDCG Curve" if normalized else 'DCG Curve')
    plt.legend()
    plt.grid(True)
    
    if show:
        path = os.path.join("movementpredictor/evaluation/plots", evalconfig.camera)
        os.makedirs(path, exist_ok=True)
        name_start = "ndcg-curve-" if normalized else "dcg-curve-"
        name_further = "squared-relevance-" if squared_relevance else "linear-relevance-"
        plt.savefig(os.path.join(path, name_start + name_further + scoring + ".png"))
        plt.show()
        plt.close()
    
    return max_dcg, max_dcg_std, min_length


def find_matching_files(search_dir: str, prefix: str) -> List[str]:
    p = Path(search_dir)
    return [f for f in p.iterdir() if f.is_file() and f.name.startswith(prefix) and f.name.endswith("all_labeled_data.json")]
    

def dcg_score_tie_aware(rels, y_score):
    if rels.shape != y_score.shape:
        log.error("labels and score lists have to be of same length.")
        exit(1)

    ranks = rankdata(-y_score, method='max')

    dcg = 0
    for rel, rank in zip(rels, ranks):
        dcg += rel/np.log2(rank+1)
    
    return dcg



def dcg_mean_and_std(trajectories_of_all_runs: List[List[PredictedTrajectory]], min_num_anomaly_frames: int, 
                     all_ids: List[str], all_group_labels: List[int], scoring: str, rels: List[int], idcg: float = None):
    dcgs = []

    for trajectories in trajectories_of_all_runs:
        scores = get_scores(trajectories, min_num_anomaly_frames, all_ids, all_group_labels, scoring)#, remove_undetected=True)

        #dcgs_temp = []
        #for i in range(10):     # important to add random noice and repeat several times becuase many samples have the same score 0 and will get a different ranking each time
         #   rng = np.random.default_rng(seed=i)  
          #  noise = rng.normal(loc=0.0, scale=1e-9, size=len(y_true))
           # scores = np.array(scores+noise, dtype=np.float64)
            #dcgs_temp.append(ndcg_score([np.array(rels)], [np.array(scores)]) if normalized else dcg_score([np.array(rels)], [np.array(scores)]))
        
        #dcgs.append(np.mean(dcgs_temp))
        score = dcg_score_tie_aware(np.array(rels), np.array(scores))
        dcgs.append(score if idcg is None else score / idcg)

    mean = np.mean(np.array(dcgs))
    std = np.std(np.array(dcgs))

    return mean, std



all_ids, all_group_labels, all_event_labels = all_predicted_ids_with_group(evalconfig.path_label_box, evalconfig.camera)
print("num total detected trajectories: ", str(len([group for group in all_group_labels if group != 0])))

label_counts = Counter(all_event_labels)
sorted_labels = sorted(label_counts.keys())
frequencies = [label_counts[label] for label in sorted_labels]
del sorted_labels[1]
del frequencies[1]

plt.figure(figsize=(8, 5))
plt.bar(sorted_labels, frequencies)
plt.xlabel("Label")
plt.ylabel("Frequency")
plt.title("Histogram of event labels")
plt.xticks(sorted_labels)  
plt.grid(axis='y')
plt.tight_layout()
path = os.path.join("movementpredictor/evaluation/plots", evalconfig.camera)
os.makedirs(path, exist_ok=True)
plt.savefig(os.path.join(path, "EventFrequencies.png"))
plt.close()


'''CREATION OF PR-CURVES PLOT'''
plt.figure(figsize=(10, 6))
model_names = ["0.5sec_MobileNet_v3_symmetric_prob", "0.5sec_MobileNet_v3_asymmetric_prob", "1sec_MobileNet_v3_symmetric_prob", "1sec_MobileNet_v3_asymmetric_prob",
               "2sec_MobileNet_v3_symmetric_prob", "2sec_MobileNet_v3_asymmetric_prob", "nearest_neighbor_analysis"]

good_class_start: int = 2
min_num_anomaly_frames: int = 5

for i, model_name in enumerate(model_names):
    path_list_predictions = find_matching_files(evalconfig.path_store_anomalies, model_name)

    trajectories_of_all_runs = []
    for path in path_list_predictions:
        trajectories_of_all_runs.append(get_trajectories(path))

    mean_and_variance_PR_curve(trajectories_of_all_runs, model_name, good_class_start, min_num_anomaly_frames, 
                               all_ids, all_group_labels, scoring="min", show=i==len(model_names)-1)


for scoring in ["med", "min", "avg"]:
    for start_class in [2, 3, 4]:
        plt.figure(figsize=(10, 6))

        for i, model_name in enumerate(model_names):
            path_list_predictions = find_matching_files(evalconfig.path_store_anomalies, model_name)

            trajectories_of_all_runs = []
            for path in path_list_predictions:
                trajectories_of_all_runs.append(get_trajectories(path))

            max_f1, max_f1_std, threshold, threshold_std, min_anomaly_frames = best_F1_scores_plot(trajectories_of_all_runs, model_name, start_class, all_ids, all_group_labels, 
                                                                               scoring=scoring, show=i==len(model_names)-1)

            print(model_name, " - ", start_class, " - " , scoring)
            print("Max F1 Score at threshold ", round(threshold, 3), " (+/-", round(threshold_std, 3), ") with mind. ", min_anomaly_frames, " frames: ", round(max_f1, 3), " (+/-", round(max_f1_std, 3), ")\n")


'''
for scoring in ["min", "med", "avg"]:
    for start_class in [2, 3, 4]:
        plt.figure(figsize=(10, 6))

        for i, model_name in enumerate(model_names):
            path_list_predictions = find_matching_files(evalconfig.path_store_anomalies, model_name)

            trajectories_of_all_runs = []
            for path in path_list_predictions:
                trajectories_of_all_runs.append(get_trajectories(path))

            PR_AUCs(trajectories_of_all_runs, model_name, start_class, all_ids, all_group_labels, scoring=scoring, show=i==len(model_names)-1)
'''

for squared_relevance in [True, False]:
    for scoring in ["min", "med", "avg"]:

        '''
        plt.figure(figsize=(10, 6))
        for i, model_name in enumerate(model_names):
            path_list_predictions = find_matching_files(evalconfig.path_store_anomalies, model_name)

            trajectories_of_all_runs = []
            for path in path_list_predictions:
                trajectories_of_all_runs.append(get_trajectories(path))

            max_dcg, max_dcg_std, min_length = dcg_curve(trajectories_of_all_runs, model_name, all_ids, all_group_labels, show=i==len(model_names)-1)
            print(model_name + " - " + scoring)
            print("squared relevance" if squared_relevance else "linear relevance")
            print("Max DCG Score at ", min_length, ": ", max_dcg, " (std: ", max_dcg_std, ")\n")
        '''

        plt.figure(figsize=(10, 6))
        for i, model_name in enumerate(model_names):
            path_list_predictions = find_matching_files(evalconfig.path_store_anomalies, model_name)

            trajectories_of_all_runs = []
            for path in path_list_predictions:
                trajectories_of_all_runs.append(get_trajectories(path))

            max_dcg, max_dcg_std, min_length = dcg_curve(trajectories_of_all_runs, model_name, all_ids, all_group_labels, scoring, 
                                                         normalized=True, show=i==len(model_names)-1, squared_relevance=squared_relevance)
            print(model_name + " - " + scoring)
            print("squared relevance" if squared_relevance else "linear relevance")
            print("Max NDCG Score at ", min_length, ": ", max_dcg, " (std: ", max_dcg_std, ")\n")



'''
def mean_and_variance_abs_frequencies(trajectories_of_all_runs: List[List[PredictedTrajectory]], model_name: str, good_class_start: int,
                                      min_num_anomaly_frames: int, show: bool = False):
    frequencies_of_all_runs = []
    for trajectory in trajectories_of_all_runs:
        frequencies_of_all_runs.append(calculate_abs_frequencies(trajectory))
    
    max_len = max([len(frequencies) for frequencies in frequencies_of_all_runs])
    for frequencies in frequencies_of_all_runs:
        last_freq = frequencies[-1]
        while len(frequencies) < max_len:
            frequencies.append(last_freq)

    frequencies_of_all_runs = np.array(frequencies_of_all_runs)
    mean = np.mean(frequencies_of_all_runs, axis=0)
    std = np.std(frequencies_of_all_runs, axis=0)
    x = np.arange(len(mean))  

    plt.plot(x, mean, label=model_name)
    plt.fill_between(x, mean-0.5*std, mean+0.5*std, alpha=0.3)

    plt.xlabel("Ranking")
    plt.ylabel("#Anomalies")
    plt.title("Start class positives: " + str(good_class_start) + ", min_anomal_detections: " + str(min_num_anomaly_frames))
    plt.legend()
    plt.grid(True)
    
    if show:
        path = os.path.join("movementpredictor/evaluation/plots", evalconfig.camera)
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, "absolute-frequencies-" + str(good_class_start) + "-" + str(min_num_anomaly_frames) + ".png"))
        plt.show()
        plt.close()
'''
'''
def calculate_abs_frequencies(trajectories: List[PredictedTrajectory]) -> List[int]:
    y_true = []
    scores = []

    for trajectory in trajectories:
        if len(trajectory.measures) < min_num_anomaly_frames or trajectory.anomaly_group == 0:
            continue

        y_true.append(1 if trajectory.obj_id in ground_truth else 0)
        sorted_measures = sorted(trajectory.measures, reverse=True)
        scores.append(sorted_measures[min_num_anomaly_frames - 1])

    y_true = np.array(y_true)
    scores = np.clip(scores, 0, np.finfo(np.float32).max)
    scores = np.array(scores, dtype=np.float32)

    sorted_indices = np.argsort(scores)[::-1]
    sorted_y_true = np.array(y_true)[sorted_indices]

    cumulative_positives = np.cumsum(sorted_y_true)
    return list(cumulative_positives)
'''

'''CREATION OF FREQUENCIES PLOT'''
'''
plt.figure(figsize=(12, 8))

model_name = "0.5sec_MobileNet_v3_symmetric_prob"
path_list_predictions = find_matching_files(evalconfig.path_store_anomalies, model_name)

trajectories_of_all_runs = []
for path in path_list_predictions:
    trajectories_of_all_runs.append(get_trajectories(path))

mean_and_variance_abs_frequencies(trajectories_of_all_runs, model_name)


model_name = "0.5sec_MobileNet_v3_asymmetric_prob"
path_list_predictions = find_matching_files(evalconfig.path_store_anomalies, model_name)

trajectories_of_all_runs = []
for path in path_list_predictions:
    trajectories_of_all_runs.append(get_trajectories(path))

mean_and_variance_abs_frequencies(trajectories_of_all_runs, model_name)


model_name = "1sec_MobileNet_v3_symmetric_prob"
path_list_predictions = find_matching_files(evalconfig.path_store_anomalies, model_name)

trajectories_of_all_runs = []
for path in path_list_predictions:
    trajectories_of_all_runs.append(get_trajectories(path))

mean_and_variance_abs_frequencies(trajectories_of_all_runs, model_name)


model_name = "1sec_MobileNet_v3_asymmetric_prob"
path_list_predictions = find_matching_files(evalconfig.path_store_anomalies, model_name)

trajectories_of_all_runs = []
for path in path_list_predictions:
    trajectories_of_all_runs.append(get_trajectories(path))

mean_and_variance_abs_frequencies(trajectories_of_all_runs, model_name)


model_name = "2sec_MobileNet_v3_symmetric_prob"
path_list_predictions = find_matching_files(evalconfig.path_store_anomalies, model_name)

trajectories_of_all_runs = []
for path in path_list_predictions:
    trajectories_of_all_runs.append(get_trajectories(path))

mean_and_variance_abs_frequencies(trajectories_of_all_runs, model_name)


model_name = "2sec_MobileNet_v3_asymmetric_prob"
path_list_predictions = find_matching_files(evalconfig.path_store_anomalies, model_name)

trajectories_of_all_runs = []
for path in path_list_predictions:
    trajectories_of_all_runs.append(get_trajectories(path))

mean_and_variance_abs_frequencies(trajectories_of_all_runs, model_name)


model_name = "nearest_neighbor_analysis"
path_list_predictions = find_matching_files(evalconfig.path_store_anomalies, model_name)

trajectories_of_all_runs = []
for path in path_list_predictions:
    trajectories_of_all_runs.append(get_trajectories(path))

mean_and_variance_abs_frequencies(trajectories_of_all_runs, model_name, show=True)
'''