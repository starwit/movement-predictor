from collections import Counter

import numpy as np
from movementpredictor.evaluation.eval_config import EvalConfig
import os
import matplotlib.pyplot as plt

from movementpredictor.evaluation.metrics_and_results import evaluation_helper, ndcg, precision_recall_f1, histograms


evalconfig = EvalConfig()



all_ids, all_group_labels, all_event_labels = evaluation_helper.all_predicted_ids_with_group(evalconfig.path_label_box, evalconfig.camera)
print("num total detected trajectories: ", str(len([group for group in all_group_labels if group != -1])))

model_names = ["0.5sec_MobileNet_v3_symmetric_prob", "0.5sec_MobileNet_v3_asymmetric_prob", "1sec_MobileNet_v3_symmetric_prob", "1sec_MobileNet_v3_asymmetric_prob",
               "2sec_MobileNet_v3_symmetric_prob", "2sec_MobileNet_v3_asymmetric_prob", "nearest_neighbor_analysis"]
model_labels = ["SYM0.5", "ASYM0.5", "SYM1", "ASYM1", "SYM2", "ASYM2", "NNC"]
#model_colors = ["mediumblue", "cornflowerblue", "darkgreen", "limegreen", "darkred", "orangered", "black"]
model_colors = ["cornflowerblue", "cornflowerblue", "darkgreen", "darkgreen", "orangered", "orangered", "black"]


'''CREATION OF HISTOGRAMS ON DATA COLLECTED FOR LABELING'''
group_colors = {
    4: 'tab:red',
    3: 'tab:orange',
    2: 'tab:green',
    1: 'tab:blue',
    0: 'tab:purple',
    -1: 'tab:gray'
}
group_annotations = ["det. mistake", "0: no interest", "1: less interest", "2: relevant", "3: high relevance", "4: dangerous"]
hg = histograms.GroupHistogram(evaluation_helper.groups, group_colors, group_annotations)
hg.plot_all_found_events(all_event_labels)
hg.plot_all_found_anomalies(all_event_labels)


for model_name in model_names:
    hg.plot_per_method(evalconfig.path_store_anomalies, model_name)

#hg.log_anomaly_event_counts_per_group(evalconfig.path_store_anomalies, model_names)
hg.summary_histogram_pmf_vs_nn(evalconfig.path_store_anomalies, pmf_name="0.5sec_MobileNet_v3_symmetric_prob", label="PMF SYM0.5")

path_store_plot = os.path.join("movementpredictor/evaluation/plots", evalconfig.camera)
os.makedirs(path_store_plot, exist_ok=True)

def _load_trajectories_for_model(model_name: str):
    paths = evaluation_helper.find_matching_files(evalconfig.path_store_anomalies, model_name)
    return [evaluation_helper.get_trajectories(p) for p in paths]

print("------------------ SCORING BASED ON SCORE^(k) ---------------------\n")

fig, ax = plt.subplots(figsize=(6, 3.5))
for i, model_name in enumerate(model_names):
    tra_runs = _load_trajectories_for_model(model_name)

    max_dcg, max_std, argmax = ndcg.ndcg_curve_initial_scoring(
        tra_runs, all_ids, all_group_labels, k=50,
        label=model_labels[i], color=model_colors[i], ax=ax, include_mistakes=True
    )
    print(f"{model_name} - score^(k): \tMax NDCG Score at {argmax} : {max_dcg} (std: {max_std})\n")

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
fig.savefig(os.path.join(path_store_plot, f"{evalconfig.camera}-ndcg-curve-initial-scoring.png"), dpi=200, bbox_inches='tight')
plt.close(fig)

print("------------------ SCORINGS BASED ON TOP-K ANOMALY POINTS PER TRAJECTORY ---------------------\n")

scorings = ["med", "avg", "weighted-avg"]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)

for ax, scoring in zip(axes, scorings):
    for i, model_name in enumerate(model_names[:-1]):  # Exclude NNC here (as you did)
        tra_runs = _load_trajectories_for_model(model_name)
        max_dcg, max_std, argmax = ndcg.ndcg_curve_topk(
            tra_runs, all_ids, all_group_labels,
            scoring=scoring, k=50, label=model_labels[i], color=model_colors[i], ax=ax, include_mistakes=True
        )
        print(f"{model_name} - {scoring}: \tMax NDCG Score at {argmax} : {max_dcg} (std: {max_std})")

# one shared legend on the right
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.875, 0.5), title=None)
fig.tight_layout()
fig.subplots_adjust(right=0.87)
fig.savefig(os.path.join(path_store_plot, f"{evalconfig.camera}-ndcg-curve-top-k-rankings-combined.png"), dpi=200, bbox_inches='tight')
plt.close(fig)

print("------------------ SCORING BASED ON TOP-PERCENTILE ANOMALY POINTS PER TRAJECTORY ---------------------\n")

scorings = ["med", "avg", "weighted-avg"]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)

for ax, scoring in zip(axes, scorings):
    for i, model_name in enumerate(model_names[:-1]):  # Exclude NNC here (as you did)
        tra_runs = _load_trajectories_for_model(model_name)
        max_dcg, max_std, best_pct = ndcg.ndcg_curve_percentil(
            tra_runs, all_ids, all_group_labels,
            scoring=scoring, k=50, label=model_labels[i], color=model_colors[i], ax=ax
        )
        print(f"{model_name} - {scoring}: \tMax NDCG Score with portion {best_pct} : {max_dcg} (std: {max_std})")

handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.875, 0.5))
fig.tight_layout()
fig.subplots_adjust(right=0.87)
fig.savefig(os.path.join(path_store_plot, f"{evalconfig.camera}-ndcg-curve-percentil-rankings-combined.png"), dpi=200, bbox_inches='tight')
plt.close(fig)


print("------------------ SCORING BASED ON WHOLE TRAJECTORY ---------------------\n")

fig, ax = plt.subplots(figsize=(6, 3.5))
for i, model_name in enumerate(model_names[:-1]):  # Exclude NNC if desired
    tra_runs = _load_trajectories_for_model(model_name)
    max_dcg, max_std, best_a = ndcg.ndcg_curve_exp_weighted_avg(
        tra_runs, all_ids, all_group_labels, k=50, label=model_labels[i], color=model_colors[i], ax=ax, include_mistakes=True
    )
    print(f"{model_name} - exp-weighted-avg: \tMax NDCG Score with a={round(best_a, 2)} : {max_dcg} (std: {max_std})")

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
fig.savefig(os.path.join(path_store_plot, f"{evalconfig.camera}-ndcg-curve-exp-weighted-scoring-top50-anomalies.png"), dpi=200, bbox_inches='tight')
plt.close(fig)


# HISTOGRAMS BEST SCORING
num_trajectories = [10, 20, 50]
fig, axes = plt.subplots(1, 3, figsize=(16, 3.5), sharey=True)
wspace = 0.1

for ax, num_tr in zip(axes, num_trajectories):
    path_list_predictions = evaluation_helper.find_matching_files(
        evalconfig.path_store_anomalies, "1sec_MobileNet_v3_asymmetric_prob"
    )

    trajectories_of_all_runs = [
        evaluation_helper.get_trajectories(path) for path in path_list_predictions
    ]

    # draw INTO the given axis
    hg.plot_best_scoring_histogram(
        trajectories_of_all_runs, weight_param=0.98, k=num_tr,
        ax=ax, show_legend=num_tr==50  # legend only once, outside
    )
    ax.set_title(f"Top {num_tr} anomaly candidates")

# one shared legend on the right
#handles, labels = axes[-1].get_legend_handles_labels()
#fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), title='Relevance')

plt.tight_layout()
plt.subplots_adjust(wspace=wspace, right=0.86)  # room for legend

out_path = os.path.join(path_store_plot, f"{evalconfig.camera}-histograms-best-scoring.png")
os.makedirs(path_store_plot, exist_ok=True)
fig.savefig(out_path, dpi=200)
plt.close(fig)

print("----------------------- AUC PR -----------------------")
scorings = ["min-exc", "avg", "exp-weighted-avg"]      # rows
start_classes = [1, 2, 3, 4]                           # columns

fig, axes = plt.subplots(3, 4, figsize=(16, 12), sharey=True)

# Nice row titles
row_titles = {
    "min-exc": r"Trajectory ranking score$^{(k)}$",
    "avg":     r"Trajectory ranking score$_{avg}^{(k)}$",
    "exp-weighted-avg": r"Trajectory ranking score$_{exp}^{(a)}$",
}

# Column titles (only on the first row)
col_titles = [f"Relevance ≥ {c}" for c in start_classes]
for c, title in enumerate(col_titles):
    axes[0, c].set_title(title)

# Draw grid
for r, scoring in enumerate(scorings):
    for c, start_class in enumerate(start_classes):
        ax = axes[r, c]
        plt.sca(ax)  # make this axes current if your plotting fn draws on the current axes

        for i, model_name in enumerate(model_names):
            # Skip NNC for non "min-exc" scoring if that's your rule
            if scoring != "min-exc" and model_name == "nearest_neighbor_analysis":
                continue

            path_list_predictions = evaluation_helper.find_matching_files(
                evalconfig.path_store_anomalies, model_name
            )
            trajectories_of_all_runs = [
                evaluation_helper.get_trajectories(path) for path in path_list_predictions
            ]

            if scoring == "exp-weighted-avg":
                precision_recall_f1.PR_AUCs_exp_weighting(
                    trajectories_of_all_runs, 
                    start_class, 
                    all_ids=all_ids, 
                    all_group_labels=all_group_labels, 
                    label=model_labels[i],
                    color=model_colors[i]
                )
            else:
                precision_recall_f1.PR_AUCs(
                    trajectories_of_all_runs,
                    start_class,
                    all_ids,
                    all_group_labels,
                    scoring=scoring,
                    label=model_labels[i],
                    color=model_colors[i]
                )

        # Gridlines per subplot
        ax.grid(True, alpha=0.3)

    # Add a centered row subtitle above the middle subplot of the row
    axes[r, 1].text(
        0.5, 1.12, row_titles.get(scorings[r], scorings[r]),
        transform=axes[r, 1].transAxes,
        ha='center', va='bottom', fontsize=12, fontweight='bold'
    )

# One legend on the right: collect unique handles/labels from all axes
handles, labels = [], []
for ax in axes.flat:
    h, l = ax.get_legend_handles_labels()
    for hh, ll in zip(h, l):
        if ll and ll not in labels:
            handles.append(hh)
            labels.append(ll)

fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.88, 0.5), frameon=False)

fig.tight_layout()
fig.subplots_adjust(right=0.87, hspace=0.35)

out_path = os.path.join(path_store_plot, f"{evalconfig.camera}-auc-pr-grid.png")
os.makedirs(path_store_plot, exist_ok=True)
fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.2)  
plt.close(fig)

start_classes = [1, 2, 3, 4]                           # columns
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
wspace = 0.1

for ax, start_class in zip(axes, start_classes):
    plt.sca(ax)  # make this axes current if your plotting fn draws on the current axes

    for i, model_name in enumerate(model_names):
        path_list_predictions = evaluation_helper.find_matching_files(
            evalconfig.path_store_anomalies, model_name
        )
        trajectories_of_all_runs = [
            evaluation_helper.get_trajectories(path) for path in path_list_predictions
        ]

        precision_recall_f1.PR_AUCs(
            trajectories_of_all_runs,
            start_class,
            all_ids,
            all_group_labels,
            scoring="min-exc",
            label=model_labels[i],
            color=model_colors[i]
        )
        
    # Gridlines per subplot
    ax.grid(True, alpha=0.3)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()

out_path = os.path.join(path_store_plot, f"{evalconfig.camera}-auc-pr-initial-scoring.png")
os.makedirs(path_store_plot, exist_ok=True)
fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.2)  
plt.close(fig)


start_classes = [1, 2, 3, 4]                           # columns
model_name = "1sec_MobileNet_v3_asymmetric_prob"

fig, ax = plt.subplots(figsize=(8, 4.5))

for i, start_class in enumerate(start_classes):
    path_list_predictions = evaluation_helper.find_matching_files(
        evalconfig.path_store_anomalies, model_name
    )
    trajectories_of_all_runs = [
        evaluation_helper.get_trajectories(path) for path in path_list_predictions
    ]

    # If your function accepts an axes, pass ax=ax; otherwise plt.sca(ax) above is enough.
    precision_recall_f1.mean_and_variance_PR_curve(
        trajectories_of_all_runs, 
        start_class, 
        all_ids=all_ids, 
        all_group_labels=all_group_labels, 
        label=f"Min rel ≥ {start_class}",
        color=group_colors[start_class],
        scoring="exp-weighted-avg",
        weight_param=0.98
    )

# Gridlines per subplot
ax.grid(True, alpha=0.3)

# One legend on the right: collect unique handles/labels from all axes
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()

out_path = os.path.join(path_store_plot, f"{evalconfig.camera}pr-curves-best-scoring.png")
os.makedirs(path_store_plot, exist_ok=True)
fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.2)  
plt.close(fig)

exit(0)


#for event_of_interest in [22, 18, 11, 12, 16]:
good_classes = [2, 3, 4]
fig, axes = plt.subplots(1, 3, figsize=(16, 3.5), sharey=True)
wspace = 0.1

for ax, good_class_start in zip(axes, good_classes):
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
    for start_class in [1, 2, 3, 4]:
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


