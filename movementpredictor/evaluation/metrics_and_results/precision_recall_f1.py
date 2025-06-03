from movementpredictor.evaluation.eval_config import EvalConfig
from typing import List, Tuple
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

from movementpredictor.evaluation.metrics_and_results import evaluation_helper


log = logging.getLogger(__name__)
evalconfig = EvalConfig()


def calculate_PR_curve(trajectories: List[evaluation_helper.PredictedTrajectory], good_class_start: int, weigth_param: float, 
                       all_ids: List[str], all_group_labels: List[int], scoring: str) -> List[Tuple[float, float]]:
    
    y_true = evaluation_helper.get_y_true(all_group_labels)
    scores = evaluation_helper.get_scores_full_trajectory(trajectories, all_ids, all_group_labels, scoring, exp_para=weigth_param)
    y_true = np.array([class_label >= good_class_start for class_label in y_true])

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)

    return precision, recall, thresholds, pr_auc


def calculate_PR_curve_top_k_points(trajectories: List[evaluation_helper.PredictedTrajectory], good_class_start: int, min_num_anomaly_frames: int, 
                       all_ids: List[str], all_group_labels: List[int], scoring: str) -> List[Tuple[float, float]]:
    
    y_true = evaluation_helper.get_y_true(all_group_labels)
    scores = evaluation_helper.get_scores_top_k(trajectories, min_num_anomaly_frames, all_ids, all_group_labels, scoring)
    y_true = np.array([class_label >= good_class_start for class_label in y_true])

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)

    return precision, recall, thresholds, pr_auc


def best_F1_scores_plot(trajectories_of_all_runs: List[List[evaluation_helper.PredictedTrajectory]], model_name: str, good_class_start: int,
            all_ids: List[str], all_group_labels: List[int], scoring: str, show: bool = False):
    best_f1 = []
    best_f1_std = []
    best_f1_threshold = []
    best_f1_threshold_std = []

    max_min_anomaly_frames = 50 if scoring == "min" else 80

    for i in range(max_min_anomaly_frames):
        best_f1_temp = []
        best_f1_threshold_temp = []

        for trajectory in trajectories_of_all_runs:
            precision, recall, thresholds, _ = calculate_PR_curve_top_k_points(trajectory, good_class_start, i+1, all_ids, all_group_labels, scoring)
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


def PR_AUC(trajectories_of_all_runs: List[List[evaluation_helper.PredictedTrajectory]], good_class_start: int, weight_param: float, 
           all_ids: List[str], all_group_labels: List[int], scoring: str):

    aucs = []
    for trajectory in trajectories_of_all_runs:
        _, _, _, pr_auc = calculate_PR_curve(trajectory, good_class_start, weight_param, all_ids, all_group_labels, scoring)
        aucs.append(pr_auc)

    mean = np.mean(aucs)
    std = np.std(aucs)
    return mean, std
    


def PR_AUCs(trajectories_of_all_runs: List[List[evaluation_helper.PredictedTrajectory]], model_name: str, good_class_start: int,
            all_ids: List[str], all_group_labels: List[int], scoring: str, show: bool = False):
    aucs = []
    aucs_std = []
    max_min_anomaly_frames = 50 if scoring == "min" else 80

    for i in range(max_min_anomaly_frames):
        aucs_temp = []
        for trajectory in trajectories_of_all_runs:
            _, _, _, pr_auc = calculate_PR_curve_top_k_points(trajectory, good_class_start, i+1, all_ids, all_group_labels, scoring)
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
        plt.close()


def mean_and_variance_PR_curve(trajectories_of_all_runs: List[List[evaluation_helper.PredictedTrajectory]], model_name: str, good_class_start: int,
                               weight_param: float, all_ids: List[str], all_group_labels: List[int], scoring: str, show: bool = False):
    pr_curves: List[np.array] = []
    for trajectory in trajectories_of_all_runs:
        precision, recall, _, _ = calculate_PR_curve(trajectory, good_class_start, weight_param, all_ids, all_group_labels, scoring)
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
    plt.title("Start class positives: " + str(good_class_start))
    plt.legend()
    plt.grid(True)
    
    if show:
        path = os.path.join("movementpredictor/evaluation/plots", evalconfig.camera)
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, "pr-curves-" + str(good_class_start) + ".png"))
        plt.show()
        plt.close()

