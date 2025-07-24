from collections import Counter
import logging
import os
from typing import List
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from movementpredictor.evaluation.eval_config import EvalConfig
from movementpredictor.evaluation.metrics_and_results import evaluation_helper
from movementpredictor.evaluation.metrics_and_results.evaluation_helper import  PredictedTrajectory, find_matching_files_top_k, get_trajectories


log = logging.getLogger(__name__)
evalconfig = EvalConfig()


class GroupHistogram:

    def __init__(self, groups, group_colors, group_annotations):
        """
        groups: dict mapping group_id -> list of values
        group_colors: dict mapping group_id -> color string
        """
        self.groups = groups
        self.key_order = sorted(groups.keys(), reverse=True)
        
        self.x_labels = []
        for k in self.key_order:
            self.x_labels.extend(sorted(groups[k], reverse=True))
        self.n_bins = len(self.x_labels)
        
        self.value2group = {v: k for k, vals in groups.items() for v in vals}
        self.colors = [group_colors[self.value2group[val]] for val in self.x_labels]
        self.group_colors = group_colors
        self.group_annotations = group_annotations


    def plot_per_method(self, path_predictions, model_name):
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)

        for idx, num_anomaly_points in enumerate([1, 2, 5, 10, 20, 50]):
            ax = axes.flat[idx]
            path_list_predictions = find_matching_files_top_k(path_predictions, model_name, num_anomaly_points)

            trajectories = []
            for path in path_list_predictions:
                trajectories.append(get_trajectories(path))

            hists = []
            for trs in trajectories:
                found_labels = np.array([tr.label for tr in trs])
                counts = [np.count_nonzero(found_labels == val) for val in self.x_labels]
                hists.append(counts)
            hists = np.array(hists)

            mean_hist = hists.mean(axis=0)
            std_hist  = hists.std(axis=0)

            x = np.arange(self.n_bins)

            ax.bar(x, mean_hist, color=self.colors, edgecolor='black', label='_nolegend_')
            ax.errorbar(x, mean_hist, yerr=std_hist, fmt='none', ecolor='black', capsize=3)

            ax.set_xticks(x)
            ax.set_xticklabels(self.x_labels, rotation='vertical')
            ax.set_xlabel('Label Value')
            if idx % 3 == 0:
                ax.set_ylabel('Average Count')
            ax.set_title(f"Min {num_anomaly_points} unusual points per trajectory")
            ax.grid(axis='y', linestyle='-', alpha=0.3)

        handles = [
            mpatches.Patch(color=self.group_colors[k], label=self.group_annotations[k])
            for k in sorted(self.group_colors.keys(), reverse=True)
        ]
        fig.legend(
            handles=handles,
            title='Groups',
            bbox_to_anchor=(1, 0.5),
            loc='center right',
            frameon=False
        )

        fig.tight_layout(rect=[0, 0, 0.89, 1.0])

        save_path = os.path.join("movementpredictor/evaluation/plots", evalconfig.camera)
        os.makedirs(save_path, exist_ok=True)
        output_filename = os.path.join(save_path, f"histogram_{model_name}.png")
        fig.savefig(output_filename)
        plt.close(fig)


    def make_plot(self, hists):
        hists = np.array(hists) 

        # mean and std over runs
        mean_hist = hists.mean(axis=0)
        std_hist  = hists.std(axis=0)

        x = np.arange(self.n_bins)
        plt.figure(figsize=(12, 6))
        plt.bar(x, mean_hist, color=self.colors, edgecolor='black', label='_nolegend_')
        plt.errorbar(x, mean_hist, yerr=std_hist, fmt='none', ecolor='black', capsize=3)

        plt.xticks(x, self.x_labels, rotation='vertical')
        plt.xlabel('Label Value')
        plt.ylabel('Average Count')

        # legend by group
        handles = [
            mpatches.Patch(color=self.group_colors[k], label=self.group_annotations[k])
            for k in sorted(self.group_colors.keys(), reverse=True)
        ]
        plt.legend(handles=handles, title='Groups', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(axis='y')


    def plot_per_method_and_num_points(self, trajectories: List[List[PredictedTrajectory]], name_model, num_anomaly_points):
        """
        label_arrays: list of 1D numpy arrays, each containing labels in x_labels universe
        """
        # compute histogram counts for each run
        hists = []
        for trs in trajectories:
            found_labels = np.array([tr.label for tr in trs])
            counts = [np.count_nonzero(found_labels == val) for val in self.x_labels]
            hists.append(counts)

        self.make_plot(hists)

        path = os.path.join("movementpredictor/evaluation/plots", evalconfig.camera)
        os.makedirs(path, exist_ok=True)

        plt.title(name_model + " with min " + str(num_anomaly_points) + " unusual points per trajectory")
        plt.savefig(os.path.join(path, "historgram_" + name_model + "_" + str(num_anomaly_points) + ".png"))
        plt.close()


    def plot_all_found_anomalies(self, all_event_labels):
        hist = [np.count_nonzero(np.array(all_event_labels) == val) for val in self.x_labels]

        break_threshold = 58

        fig, (ax_upper, ax_lower) = plt.subplots(
            2, 1, figsize=(12, 6),
            sharex=True,
            gridspec_kw={'height_ratios': [1, 3]}
        )

        x = np.arange(self.n_bins)

        ax_upper.bar(x, hist, color=self.colors, edgecolor='black')
        ax_lower.bar(x, hist, color=self.colors, edgecolor='black')

        ax_upper.set_ylim(break_threshold, np.max(hist) * 1.1)
        ax_lower.set_ylim(0, break_threshold)

        ax_upper.spines['bottom'].set_visible(False)
        ax_lower.spines['top'].set_visible(False)

        ax_upper.tick_params(axis='x', which='both',
                         bottom=False, labelbottom=False)
        ax_lower.tick_params(axis='x', which='both',
                            top=False)

        d = .01  
        
        kwargs = dict(transform=ax_upper.transAxes, color='k', clip_on=False, linewidth=0.5)
        ax_upper.plot((-d, +d), (-d*2, +d*2), **kwargs)
        ax_upper.plot((1-d, 1+d), (-d*2, +d*2), **kwargs)

        kwargs.update(transform=ax_lower.transAxes)
        ax_lower.plot((-d, +d), (1-d*2, 1+d*2), **kwargs)
        ax_lower.plot((1-d, 1+d), (1-d*2, 1+d*2), **kwargs)

        plt.xticks(x, self.x_labels, rotation='vertical')
        ax_lower.set_xlabel('Label Value')
        ax_lower.set_ylabel('Count')

        fig.suptitle("Histogram of Event labels")

        # legend by group
        handles = [
            mpatches.Patch(color=self.group_colors[k], label=self.group_annotations[k])
            for k in sorted(self.group_colors.keys(), reverse=True)
        ]
        plt.legend(handles=handles, title='Groups', bbox_to_anchor=(1.02, 1), loc='upper left')

        fig.tight_layout()#rect=[0, 0, 1, 0.98])
        
        path = os.path.join("movementpredictor/evaluation/plots", evalconfig.camera)
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, "histogram_total_events.png"))
        plt.close()


    def plot_best_scoring_histogram(self, trajectories_of_all_runs: List[List[evaluation_helper.PredictedTrajectory]], weight_param: float, k=50):
        hists = []

        for trajectories in trajectories_of_all_runs:
            trajectories_with_score = []

            for trajectory in trajectories:
                score = evaluation_helper.score_trajectory(trajectory, "exp-weighted-avg", exp_para=weight_param)
                trajectories_with_score.append([trajectory, score])

            trajectories_with_score.sort(key=lambda x: x[1], reverse=True)
            trs = [tr[0] for tr in trajectories_with_score[:k]]
            found_labels = np.array([tr.label for tr in trs])
            counts = [np.count_nonzero(found_labels == val) for val in self.x_labels]
            hists.append(counts)
        
        self.make_plot(hists)

        path = os.path.join("movementpredictor/evaluation/plots", evalconfig.camera)
        os.makedirs(path, exist_ok=True)

        plt.title("found anomalies in top " + str(k) + " detections of best scoring method")
        plt.savefig(os.path.join(path, "historgram_best_scoring-" + str(k) + ".png"))
        plt.close()
        


