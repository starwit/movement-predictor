import os
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from movementpredictor.evaluation.eval_config import EvalConfig
from movementpredictor.evaluation.metrics_and_results import evaluation_helper
from movementpredictor.evaluation.metrics_and_results.evaluation_helper import  PredictedTrajectory, find_matching_files_top_k, get_trajectories


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
    

    def log_anomaly_event_counts_per_group(self, path_predictions, model_names):
        """
        Logs the counts of found anomaly events per group for each method, along with their standard deviation.

        :param path_predictions: Path to predictions.
        :param model_names: List of model names to evaluate.
        """
        for model_name in model_names:
            print(f"Processing model: {model_name}")
            group_counts = {group_id: [] for group_id in self.groups.keys()}

            for num_anomaly_points in [1, 2, 5, 10, 20, 50]:
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
                std_hist = hists.std(axis=0)

                for group_id, group_values in self.groups.items():
                    group_indices = [self.x_labels.index(val) for val in group_values if val in self.x_labels]
                    group_mean = sum(mean_hist[idx] for idx in group_indices)
                    group_std = sum(std_hist[idx] for idx in group_indices)
                    group_counts[group_id].append((group_mean, group_std))

            for group_id, counts in group_counts.items():
                print(f"Group {group_id}:")
                for idx, ((mean, std), num_anomaly_points) in enumerate(zip(counts, [1, 2, 5, 10, 20, 50])):
                    print(f"  Anomaly Points: {num_anomaly_points}, Mean: {mean:.2f}, Std Dev: {std:.2f}")


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
                ax.set_ylabel('Anomaly Count')
            ax.set_title(f"Min {num_anomaly_points} unusual points per trajectory")
            ax.grid(axis='y', linestyle='-', alpha=0.3)

        handles = [
            mpatches.Patch(color=self.group_colors[k], label=self.group_annotations[k+1])
            for k in sorted(self.group_colors.keys(), reverse=True)
        ]
        fig.legend(
            handles=handles,
            title='Relevance',
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

    
    def summary_histogram_pmf_vs_nn(self, path_predictions, pmf_name, label, nn_name="nearest_neighbor_analysis"):
        fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharey=True)

        model_titles = [label, "Nearest Neighbor-based Clustering"]  # Row labels

        for row_idx, model_name in enumerate([pmf_name, nn_name]):
            # Position the row titles above the subplots
            y_pos = 0.96 - row_idx * 0.49  # adjust row vertical position
            fig.text(0.5, y_pos, model_titles[row_idx],
                    ha='center', va='center', fontsize=14, fontweight='bold')

            for col_idx, num_anomaly_points in enumerate([1, 10, 50]):
                ax = axes[row_idx, col_idx]
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
                std_hist = hists.std(axis=0)

                x = np.arange(self.n_bins)

                ax.bar(x, mean_hist, color=self.colors, edgecolor='black', label='_nolegend_')
                ax.errorbar(x, mean_hist, yerr=std_hist, fmt='none', ecolor='black', capsize=3)

                ax.set_xticks(x)
                ax.set_xticklabels(self.x_labels, rotation='vertical')
                ax.set_xlabel('Label Value')

                if col_idx == 0:
                    ax.set_ylabel('Anomaly Count')

                # Only the "Top 50..." part in subplot titles
                if row_idx == 0:
                    ax.set_title(fr"Top 50 of score$^{{({num_anomaly_points})}}$", fontsize=14)
                ax.grid(axis='y', linestyle='-', alpha=0.3)

        handles = [
            mpatches.Patch(color=self.group_colors[k], label=self.group_annotations[k+1])
            for k in sorted(self.group_colors.keys(), reverse=True)
        ]
        fig.legend(
            handles=handles,
            title='Relevance',
            bbox_to_anchor=(1, 0.5),
            loc='center right',
            frameon=False
        )

        #fig.tight_layout(rect=[0, 0, 0.89, 0.9])  # reserve space for row titles
        plt.subplots_adjust(wspace=0.05, hspace=0.4, top=0.9, right=0.88, left=0.05)  # more space between rows and from top

        save_path = os.path.join("movementpredictor/evaluation/plots", evalconfig.camera)
        os.makedirs(save_path, exist_ok=True)
        output_filename = os.path.join(save_path, f"histogram_PMFvsNNC.png")
        fig.savefig(output_filename)
        plt.close(fig)


    def plot_all_found_events(self, all_event_labels):
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
            mpatches.Patch(color=self.group_colors[k], label=self.group_annotations[k+1])
            for k in sorted(self.group_colors.keys(), reverse=True)
        ]
        plt.legend(handles=handles, title='Relevance', bbox_to_anchor=(1.02, 1), loc='upper left')

        fig.tight_layout()#rect=[0, 0, 1, 0.98])
        
        path = os.path.join("movementpredictor/evaluation/plots", evalconfig.camera)
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, "histogram_total_events.png"))
        plt.close()


    def plot_all_found_anomalies(self, all_event_labels):
        hist = [np.count_nonzero(np.array(all_event_labels) == val) for val in self.x_labels if val != 0]
        print(f"found non-false-positive (label!= 0): {np.sum(hist)}")

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(self.n_bins-1)

        colors = self.colors[:-2] + [self.colors[-1]]
        ax.bar(x, hist, color=colors, edgecolor='black')

        ax.set_xticks(x)
        ax.set_xticklabels(self.x_labels[:-2] + [-1], rotation='vertical')
        ax.set_xlabel('Label Value')
        ax.set_ylabel('Count')
        ax.grid(axis='y')
        ax.set_ylim(0, 55)
        #ax.set_title("Histogram of Event Labels")

        # legend by group
        handles = [
            mpatches.Patch(color=self.group_colors[k], label=self.group_annotations[k+1])
            for k in sorted(self.group_colors.keys(), reverse=True)
        ]
        ax.legend(handles=handles, title='Relevance', bbox_to_anchor=(1.02, 1), loc='upper left')

        fig.tight_layout()
        
        path = os.path.join("movementpredictor/evaluation/plots", evalconfig.camera)
        os.makedirs(path, exist_ok=True)
        fig.savefig(os.path.join(path, "histogram_total_anomalies.png"))
        plt.close(fig)


    def plot_best_scoring_histogram(self, trajectories_of_all_runs, weight_param: float, k=50, ax=None, show_legend=True):
        if ax is None:
            ax = plt.gca()

        hists = []
        for trajectories in trajectories_of_all_runs:
            trajectories_with_score = []
            for trajectory in trajectories:
                score = evaluation_helper.score_trajectory(trajectory, "exp-weighted-avg", exp_para=weight_param)
                trajectories_with_score.append((trajectory, score))

            trajectories_with_score.sort(key=lambda x: x[1], reverse=True)
            trs = [tr[0] for tr in trajectories_with_score[:k]]
            found_labels = np.array([tr.label for tr in trs])
            counts = [np.count_nonzero(found_labels == val) for val in self.x_labels]
            hists.append(counts)

        self.make_plot(hists, ax=ax, show_legend=show_legend)


    def make_plot(self, hists, ax=None, show_legend=True):
        if ax is None:
            ax = plt.gca()

        hists = np.array(hists)
        mean_hist = hists.mean(axis=0)
        std_hist  = hists.std(axis=0)

        x = np.arange(self.n_bins)
        ax.bar(x, mean_hist, color=self.colors, edgecolor='black', label='_nolegend_')
        ax.errorbar(x, mean_hist, yerr=std_hist, fmt='none', ecolor='black', capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(self.x_labels, rotation='vertical')
        ax.set_xlabel('Label Value')
        ax.set_ylabel('Anomaly Count')
        ax.grid(axis='y')

        if show_legend:
            handles = [
                mpatches.Patch(color=self.group_colors[k], label=self.group_annotations[k+1])
                for k in sorted(self.group_colors.keys(), reverse=True)
            ]
            ax.legend(handles=handles, title='Relevance', loc='upper left', bbox_to_anchor=(1.02, 1))
        


