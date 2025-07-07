from movementpredictor.evaluation.eval_config import EvalConfig
from typing import List
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score, dcg_score
from scipy.stats import rankdata

from movementpredictor.evaluation.metrics_and_results import evaluation_helper


log = logging.getLogger(__name__)
evalconfig = EvalConfig()


def ndcg_curve(trajectories_of_all_runs: List[List[evaluation_helper.PredictedTrajectory]], model_name: str, all_ids: List[str], 
              all_group_labels: List[int], scoring: str, k=50, weight_factor=None):
    curve_mean = []
    curve_std = []
    rels = evaluation_helper.get_rels(all_group_labels)

    max_min_length = 50 if scoring == "min_exc" or scoring == "" else 240
    max_ndcg, max_ndcg_std, min_length = 0, 0, 0

    for i in range(max_min_length):
        ndcg_mean, ndcg_std = ndcg_mean_and_std_top_k(trajectories_of_all_runs, i+1, all_ids, all_group_labels, scoring, rels, weight_factor=weight_factor, k=k)
        curve_mean.append(ndcg_mean)
        curve_std.append(ndcg_std)

        if ndcg_mean > max_ndcg:
            max_ndcg = ndcg_mean
            max_ndcg_std = ndcg_std
            min_length = i
    
    curve_mean = np.array(curve_mean)
    curve_std = np.array(curve_std)
    plt.plot(range(1, max_min_length+1), curve_mean, label=model_name)
    plt.fill_between(range(1, max_min_length+1), curve_mean-0.5*curve_std, curve_mean+0.5*curve_std, alpha=0.3)
    
    plt.xlabel('min length of anomal trajectory')
    plt.ylabel('NDCG')
    plt.title("NDCG Curve")
    plt.legend()
    plt.grid(True)
    
    return max_ndcg, max_ndcg_std, min_length


def ndcg_curve_percentil(trajectories_of_all_runs: List[List[evaluation_helper.PredictedTrajectory]], model_name: str, all_ids: List[str], 
              all_group_labels: List[int], scoring: str, k=50, weight_factor=None):
    curve_mean = []
    curve_std = []
    rels = evaluation_helper.get_rels(all_group_labels)

    max_ndcg, max_ndcg_std, best_percentil = 0, 0, 0
    percentils = np.round(np.linspace(0.01, 0.99, 99), 2)

    for percentil in percentils:
        ndcg_mean, ndcg_std = ndcg_mean_and_std_percentil(trajectories_of_all_runs, percentil, all_ids, all_group_labels, scoring, rels, k=k)
        curve_mean.append(ndcg_mean)
        curve_std.append(ndcg_std)

        if ndcg_mean > max_ndcg:
            max_ndcg = ndcg_mean
            max_ndcg_std = ndcg_std
            best_percentil = percentil
    
    curve_mean = np.array(curve_mean)
    curve_std = np.array(curve_std)
    plt.plot(percentils, curve_mean, label=model_name)
    plt.fill_between(percentils, curve_mean-0.5*curve_std, curve_mean+0.5*curve_std, alpha=0.3)
    
    plt.xlabel('Percentil')
    plt.ylabel('NDCG')
    plt.title("NDCG Curve")
    plt.legend()
    plt.grid(True)
    
    return max_ndcg, max_ndcg_std, best_percentil


def ndcg_curve_exp_weighted_avg(trajectories_of_all_runs: List[List[evaluation_helper.PredictedTrajectory]], model_name: str, all_ids: List[str], 
              all_group_labels: List[int], k=50):
    curve_mean = []
    curve_std = []
    rels = evaluation_helper.get_rels(all_group_labels)
    max_ndcg, max_ndcg_std, best_param = 0, 0, 0
    weight_params = np.linspace(0.01, 1.5, 150)

    for weight_param in weight_params:
        ndcg_mean, ndcg_std = ndcg_mean_and_std_full_tr_scoring(trajectories_of_all_runs, all_ids, all_group_labels, scoring="exp-weighted-avg", rels=rels, k=k, weight_param=weight_param)
        curve_mean.append(ndcg_mean)
        curve_std.append(ndcg_std)

        if ndcg_mean > max_ndcg:
            max_ndcg = ndcg_mean
            max_ndcg_std = ndcg_std
            best_param = weight_param
    
    curve_mean = np.array(curve_mean)
    curve_std = np.array(curve_std)
    plt.plot(weight_params, curve_mean, label=model_name)
    plt.fill_between(weight_params, curve_mean-0.5*curve_std, curve_mean+0.5*curve_std, alpha=0.3)
    
    plt.xlabel('exponential weight parameter')
    plt.xlim(np.min(weight_params), np.max(weight_params))
    plt.ylabel('NDCG')
    plt.title("NDCG Curve")
    plt.legend()
    plt.grid(True)
    
    return max_ndcg, max_ndcg_std, best_param
    

def dcg_score_tie_aware(rels, y_score):
    if rels.shape != y_score.shape:
        log.error("labels and score lists have to be of same length.")
        exit(1)

    ranks = rankdata(-y_score, method='max')

    dcg = 0
    for rel, rank in zip(rels, ranks):
        dcg += rel/np.log2(rank+1)
    
    return dcg



def dcg_mean_and_std(trajectories_of_all_runs: List[List[evaluation_helper.PredictedTrajectory]], min_num_anomaly_frames: int, 
                     all_ids: List[str], all_group_labels: List[int], scoring: str, rels: List[int], idcg: float = None):
    dcgs = []

    for trajectories in trajectories_of_all_runs:
        scores = evaluation_helper.get_scores_top_k(trajectories, min_num_anomaly_frames, all_ids, all_group_labels, scoring)#, remove_undetected=True)

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



def ndcg_mean_and_std_top_k(trajectories_of_all_runs: List[List[evaluation_helper.PredictedTrajectory]], min_num_anomaly_frames: int, 
                     all_ids: List[str], all_group_labels: List[int], scoring: str, rels: List[int], weight_factor = None, k=50):
    ndcgs = []
    for trajectories in trajectories_of_all_runs:
        if weight_factor is None:
            scores = evaluation_helper.get_scores_top_k(trajectories, min_num_anomaly_frames, all_ids, all_group_labels, scoring)
        else:
            scores = evaluation_helper.get_scores_num_trajectories_based(trajectories, k, min_num_anomaly_frames, all_ids, all_group_labels, weight_length=weight_factor)
        
        ndcgs.append(ndcg_score([np.array(rels)], [np.array(scores)], k=k))

    mean = np.mean(np.array(ndcgs))
    std = np.std(np.array(ndcgs))

    return mean, std


def ndcg_mean_and_std_full_tr_scoring(trajectories_of_all_runs: List[List[evaluation_helper.PredictedTrajectory]], all_ids: List[str], all_group_labels: List[int], 
                                      scoring: str, rels: List[int], k=50, weight_param=None):
    ndcgs = []
    for trajectories in trajectories_of_all_runs:
        scores = evaluation_helper.get_scores_full_trajectory(trajectories, all_ids, all_group_labels, scoring, exp_para=weight_param)
        ndcgs.append(ndcg_score([np.array(rels)], [np.array(scores)], k=k))

    mean = np.mean(np.array(ndcgs))
    std = np.std(np.array(ndcgs))

    return mean, std


def ndcg_mean_and_std_percentil(trajectories_of_all_runs: List[List[evaluation_helper.PredictedTrajectory]], percentil: int, all_ids: List[str], all_group_labels: List[int], 
                                      scoring: str, rels: List[int], k=50):
    ndcgs = []
    for trajectories in trajectories_of_all_runs:
        scores = evaluation_helper.get_scores_top_k(trajectories, percentil, all_ids, all_group_labels, scoring)
        ndcgs.append(ndcg_score([np.array(rels)], [np.array(scores)], k=k))

    mean = np.mean(np.array(ndcgs))
    std = np.std(np.array(ndcgs))

    return mean, std



def macro_ndcg(all_rels, all_scores, all_event_labels, k=50):
    ...
    # ist dumm, aber etwas, das klassenimbalance berücksichtigt wird benötigt