from collections import Counter
from movementpredictor.evaluation.eval_config import EvalConfig
import logging
import os
import matplotlib.pyplot as plt

from movementpredictor.evaluation.metrics_and_results import evaluation_helper, dcg_and_ndcg, precision_recall_f1, histograms


log = logging.getLogger(__name__)
evalconfig = EvalConfig()



all_ids, all_group_labels, all_event_labels = evaluation_helper.all_predicted_ids_with_group(evalconfig.path_label_box, evalconfig.camera)
print("num total detected trajectories: ", str(len([group for group in all_group_labels if group != 0])))

model_names = ["0.5sec_MobileNet_v3_symmetric_prob", "0.5sec_MobileNet_v3_asymmetric_prob", "1sec_MobileNet_v3_symmetric_prob", "1sec_MobileNet_v3_asymmetric_prob",
               "2sec_MobileNet_v3_symmetric_prob", "2sec_MobileNet_v3_asymmetric_prob", "nearest_neighbor_analysis"]



'''CREATION OF HISTOGRAMS ON DATA COLLECTED FOR LABELING'''
group_colors = {
    5: 'tab:red',
    4: 'tab:orange',
    3: 'tab:green',
    2: 'tab:blue',
    1: 'tab:purple',
    0: 'tab:gray'
}
group_annotations = ["det. mistake", "no interest & FP", "less interest", "interesting", "high interest", "dangerous"]
hg = histograms.GroupHistogram(evaluation_helper.groups, group_colors, group_annotations)
hg.plot_all_found_anomalies(all_event_labels)

#for num_anomaly_points in [1, 2, 5, 10, 20, 50]:
 #   for model_name in model_names:

  #      path_list_predictions = evaluation_helper.find_matching_files_top_k(evalconfig.path_store_anomalies, model_name, num_anomaly_points)

   #     trajectories_of_all_runs = []
    #    for path in path_list_predictions:
     #       trajectories_of_all_runs.append(evaluation_helper.get_trajectories(path))

      #  hg.plot_per_method(trajectories_of_all_runs, model_name, num_anomaly_points)


for model_name in model_names:
    hg.plot_per_method(evalconfig.path_store_anomalies, model_name)


'''CREATION OF NDCG-CURVES PLOTS'''
path_store_plot = os.path.join("movementpredictor/evaluation/plots", evalconfig.camera)
os.makedirs(path_store_plot, exist_ok=True)


# TOP-K SCORING
print("------------------ SCORING BASED ON TOP-K ANOMALY POINTS PER TRAJECTORY ---------------------\n")
'''
for num_trajectories in [50]:
    print("TOP ", num_trajectories, "ANOMALIES")
    for scoring in ["min-exc", "min-inc", "med", "avg", "weighted-avg"]: #"exp-weighted-avg"]:

        plt.figure(figsize=(10, 6))
        for i, model_name in enumerate(model_names):
            if model_name == "nearest_neighbor_analysis" and scoring != "min-exc" and scoring != "min-inc" and scoring != "med":
                continue
            path_list_predictions = evaluation_helper.find_matching_files(evalconfig.path_store_anomalies, model_name)

            trajectories_of_all_runs = []
            for path in path_list_predictions:
                trajectories_of_all_runs.append(evaluation_helper.get_trajectories(path))

            max_dcg, max_dcg_std, min_length = dcg_and_ndcg.ndcg_curve_class_imbalance(trajectories_of_all_runs, model_name, all_ids, all_group_labels, 
                                                                                       all_event_labels, scoring, k=num_trajectories)
            print(model_name + " - balanced - " + scoring + ": \tmax sum balanced NDCG Score at ", min_length, ": ", max_dcg, " (std: ", max_dcg_std, ")")
        
        plt.savefig(os.path.join(path_store_plot, "ndcg-curve-balanced-top-" + str(num_trajectories) + "anomalies-" + scoring + ".png"))
        plt.show()
        plt.close()
    print()
'''

for num_trajectories in [50]:
    print("TOP ", num_trajectories, "ANOMALIES")
    for scoring in ["min-exc", "med", "avg", "weighted-avg"]: #"exp-weighted-avg"]:

        plt.figure(figsize=(10, 6))
        for i, model_name in enumerate(model_names):
            if model_name == "nearest_neighbor_analysis" and scoring != "min-exc" and scoring != "min-inc" and scoring != "med":
                continue
            path_list_predictions = evaluation_helper.find_matching_files(evalconfig.path_store_anomalies, model_name)

            trajectories_of_all_runs = []
            for path in path_list_predictions:
                trajectories_of_all_runs.append(evaluation_helper.get_trajectories(path))

            max_dcg, max_dcg_std, min_length = dcg_and_ndcg.ndcg_curve(trajectories_of_all_runs, model_name, all_ids, all_group_labels, scoring, 
                                                        k=num_trajectories)
            print(model_name + " - " + scoring + ": \tMax NDCG Score at ", min_length, ": ", max_dcg, " (std: ", max_dcg_std, ")")
        
        plt.savefig(os.path.join(path_store_plot, "ndcg-curve-top-" + str(num_trajectories) + "anomalies-" + scoring + ".png"))
        plt.show()
        plt.close()
    print()


print("------------------ SCORING BASED ON TOP-PERCENTIL ANOMALY POINTS PER TRAJECTORY ---------------------\n")
#for num_trajectories in [20, 50]:
for scoring in ["min-exc", "med", "avg", "weighted-avg", "exp-weighted-avg"]:

    plt.figure(figsize=(10, 6))
    for i, model_name in enumerate(model_names):
        path_list_predictions = evaluation_helper.find_matching_files(evalconfig.path_store_anomalies, model_name)

        trajectories_of_all_runs = []
        for path in path_list_predictions:
            trajectories_of_all_runs.append(evaluation_helper.get_trajectories(path))

        max_dcg, max_dcg_std, best_perentil = dcg_and_ndcg.ndcg_curve_percentil(trajectories_of_all_runs, model_name, all_ids, all_group_labels, scoring)
        print(model_name + " - " + scoring + ": \tMax NDCG Score with percentil ", best_perentil, ": ", max_dcg, " (std: ", max_dcg_std, ")")
    
    #plt.savefig(os.path.join(path_store_plot, "ndcg-curve-k=" + str(num_trajectories) + name_further + scoring + ".png"))
    plt.savefig(os.path.join(path_store_plot, "ndcg-curve-percentil-"+ scoring + ".png"))
    plt.show()
    plt.close()
print()


# SCORING BASED ON WHOLE TRAJECTORY
print("------------------ SCORING BASED ON WHOLE TRAJECTORY ---------------------\n")
for num_trajectories in [50]:
    print("TOP ", num_trajectories, "ANOMALIES")
    for scoring in ["avg", "weighted-avg"]:

        plt.figure(figsize=(10, 6))
        for i, model_name in enumerate(model_names):
            path_list_predictions = evaluation_helper.find_matching_files(evalconfig.path_store_anomalies, model_name)

            trajectories_of_all_runs = []
            for path in path_list_predictions:
                trajectories_of_all_runs.append(evaluation_helper.get_trajectories(path))

            rels = evaluation_helper.get_rels(all_group_labels)
            max_dcg, max_dcg_std = dcg_and_ndcg.ndcg_mean_and_std_full_tr_scoring(trajectories_of_all_runs, all_ids, all_group_labels, scoring, rels, k=num_trajectories)
            print(model_name + " - " + scoring + ": \tMax NDCG Score = ", max_dcg, " (std: ", max_dcg_std, ")")
    print()

best_params_per_method = []

for num_trajectories in [50]:
    print("TOP ", num_trajectories, "ANOMALIES")
    plt.figure(figsize=(10, 6))
    temp = []
    for i, model_name in enumerate(model_names):
        path_list_predictions = evaluation_helper.find_matching_files(evalconfig.path_store_anomalies, model_name)

        trajectories_of_all_runs = []
        for path in path_list_predictions:
            trajectories_of_all_runs.append(evaluation_helper.get_trajectories(path))

        max_dcg, max_dcg_std, best_exp_param = dcg_and_ndcg.ndcg_curve_exp_weighted_avg(trajectories_of_all_runs, model_name, all_ids, all_group_labels, k=num_trajectories)
        temp.append(best_exp_param)
        print(model_name + " - exp-weighted-avg: \tMax NDCG Score with a=", round(best_exp_param, 2), ": ", max_dcg, " (std: ", max_dcg_std, ")")
    best_params_per_method.append(temp)

    plt.savefig(os.path.join(path_store_plot, "ndcg-curve-exp-weighted-scoring-top" + str(num_trajectories) + "anomalies-" + scoring + ".png"))
    plt.show()
    plt.close()
    print()


# HISTOGRAMS BEST SCORING

#for num_trajectories in [10, 20, 50, 100]:
 #   path_list_predictions = evaluation_helper.find_matching_files(evalconfig.path_store_anomalies, "0.5sec_MobileNet_v3_symmetric_prob")

  #  trajectories_of_all_runs = []
   # for path in path_list_predictions:
    #    trajectories_of_all_runs.append(evaluation_helper.get_trajectories(path))

    #hg.plot_best_scoring_histogram(trajectories_of_all_runs, weight_param=best_params_per_method[-1][-2], k=num_trajectories)


'''CREATION OF PR-CURVES PLOT'''
plt.figure(figsize=(10, 6))
print("----------------------- AUC PR -----------------------")
for scoring in ["min-exc", "avg"]:
    for start_class in [2, 3, 4, 5]:
        plt.figure(figsize=(10, 6))

        for i, model_name in enumerate(model_names):
            path_list_predictions = evaluation_helper.find_matching_files(evalconfig.path_store_anomalies, model_name)

            trajectories_of_all_runs = []
            for path in path_list_predictions:
                trajectories_of_all_runs.append(evaluation_helper.get_trajectories(path))

            precision_recall_f1.PR_AUCs(trajectories_of_all_runs, model_name, start_class, all_ids, all_group_labels, scoring=scoring, show=i==len(model_names)-1)


exit(0)

#for event_of_interest in [22, 18, 11, 12, 16]:
for good_class_start in [2, 3, 4, 5]:
    for i, (model_name, weight_param) in enumerate(zip(model_names, best_params_per_method[-1])):
        path_list_predictions = evaluation_helper.find_matching_files(evalconfig.path_store_anomalies, model_name)

        trajectories_of_all_runs = []
        for path in path_list_predictions:
            trajectories_of_all_runs.append(evaluation_helper.get_trajectories(path))

        precision_recall_f1.mean_and_variance_PR_curve(trajectories_of_all_runs, model_name, good_class_start, weight_param, 
                                all_ids, all_group_labels, scoring="exp-weighted-avg", show=i==len(model_names)-1)
        auc, std_auc = precision_recall_f1.PR_AUC(trajectories_of_all_runs, good_class_start, weight_param, 
                                all_ids, all_group_labels, scoring="exp-weighted-avg")
        print(model_name + " - positives starting at class ", good_class_start, ": \tAUC PR = ", auc, " (std: ", std_auc, ")")
print()

for scoring in ["min-exc", "avg"]:
    for start_class in [2, 3, 4, 5]:
        plt.figure(figsize=(10, 6))

        for i, model_name in enumerate(model_names):
            path_list_predictions = evaluation_helper.find_matching_files(evalconfig.path_store_anomalies, model_name)

            trajectories_of_all_runs = []
            for path in path_list_predictions:
                trajectories_of_all_runs.append(evaluation_helper.get_trajectories(path))

            max_f1, max_f1_std, threshold, threshold_std, min_anomaly_frames = precision_recall_f1.best_F1_scores_plot(trajectories_of_all_runs, model_name, start_class, all_ids, all_group_labels, 
                                                                               scoring=scoring, show=i==len(model_names)-1)

            print(model_name, " - ", start_class, " - " , scoring)
            print("Max F1 Score at threshold ", round(threshold, 3), " (+/-", round(threshold_std, 3), ") with mind. ", min_anomaly_frames, " frames: ", round(max_f1, 3), " (+/-", round(max_f1_std, 3), ")\n")



