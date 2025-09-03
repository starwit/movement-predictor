from typing import Callable, Iterable, List, Tuple, Optional
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score

from movementpredictor.evaluation.metrics_and_results import evaluation_helper

log = logging.getLogger(__name__)

PredTrajs = List[evaluation_helper.PredictedTrajectory]


# ----------------------------- Small utilities -----------------------------

def _pick_linestyle(label: Optional[str]) -> str:
    """Choose a consistent linestyle from the label prefix."""
    if not label:
        return "-"
    if label.startswith("NNC"):
        return ":"
    if label.startswith("SYM"):
        return "-"
    return "--"  # default for ASYM, etc.


def _plot_curve(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    label: Optional[str] = None,
    color: Optional[str] = None,
    mark_indices: Optional[List[int]] = None,
    title: Optional[str] = None,
    xlabel: str = r"$k$ ($k$-th largest per-detection score)",
    ylabel: str = "NDCG@50",
    add_legend: bool = False,
    legend_loc: str = "center left",
    legend_bbox_to_anchor: Tuple[float, float] = (1, 0.5),
    alpha_band: float = 0.2,
):
    """Generic plotting routine shared by all curves."""
    style = _pick_linestyle(label)
    ax.plot(x, mean, label=label, color=color, linestyle=style)
    ax.fill_between(x, mean - 0.5 * std, mean + 0.5 * std, alpha=alpha_band, color=color)

    if mark_indices:
        # Clamp to valid range, convert to x positions
        valid = [i for i in mark_indices if 0 <= i < len(x)]
        ax.scatter(x[valid], mean[valid], marker="x", s=70, color=color, zorder=5)

    if title:
        ax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

    if add_legend:
        ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)


def _print_keypoints(curve_mean: np.ndarray, curve_std: np.ndarray):
    """Print the same ‘score1/2/5/10/20/50’ snapshot you had before."""
    idx = {1: 0, 2: 1, 5: 4, 10: 9, 20: 19, 50: 49}
    def safe(arr, i): return float(arr[i]) if i < len(arr) else float("nan")
    print(
        "score1:", round(safe(curve_mean, idx[1]), 2),
        ", score2:", round(safe(curve_mean, idx[2]), 2),
        ", score5:", round(safe(curve_mean, idx[5]), 2),
        ", score10:", round(safe(curve_mean, idx[10]), 2),
        ", score20:", round(safe(curve_mean, idx[20]), 2),
        ", score50:", round(safe(curve_mean, idx[50]), 2),
    )
    print(
        "stddev:", round(safe(curve_std, idx[1]), 2),
        ", stddev:", round(safe(curve_std, idx[2]), 2),
        ", stddev:", round(safe(curve_std, idx[5]), 2),
        ", stddev:", round(safe(curve_std, idx[10]), 2),
        ", stddev:", round(safe(curve_std, idx[20]), 2),
        ", stddev:", round(safe(curve_std, idx[50]), 2),
    )


# ----------------------------- NDCG helpers -----------------------------

def _ndcg_mean_std_over_runs(scores_per_run: List[np.ndarray], rels: np.ndarray, k: int) -> Tuple[float, float]:
    """Compute mean/std of NDCG over multiple runs given per-run score vectors."""
    ndcgs = [ndcg_score([rels], [scores], k=k) for scores in scores_per_run]
    ndcgs = np.array(ndcgs, dtype=float)
    return float(np.mean(ndcgs)), float(np.std(ndcgs))


def ndcg_mean_and_std_top_k(
    trajectories_of_all_runs: List[PredTrajs],
    min_num_anomaly_frames: int,
    all_ids: List[str],
    all_group_labels: List[int],
    scoring: str,
    rels: List[int],
    k: int = 50,
    include_mistakes: bool = False,
) -> Tuple[float, float]:
    scores_per_run = []
    for trajectories in trajectories_of_all_runs:
        scores = evaluation_helper.get_scores(
            trajectories, all_ids, all_group_labels, min_num_anomaly_frames=min_num_anomaly_frames, scoring=scoring, include_mistakes=include_mistakes
        )
        scores_per_run.append(np.array(scores))
    return _ndcg_mean_std_over_runs(scores_per_run, np.array(rels), k)


def ndcg_mean_and_std_full_tr_scoring(
    trajectories_of_all_runs: List[PredTrajs],
    all_ids: List[str],
    all_group_labels: List[int],
    scoring: str,
    rels: List[int],
    k: int = 50,
    weight_param: Optional[float] = None,
    include_mistakes: bool = False,
) -> Tuple[float, float]:
    scores_per_run = []
    for trajectories in trajectories_of_all_runs:
        scores = evaluation_helper.get_scores_full_trajectory(
            trajectories, all_ids, all_group_labels, scoring, exp_para=weight_param, include_mistakes=include_mistakes
        )
        scores_per_run.append(np.array(scores))
    return _ndcg_mean_std_over_runs(scores_per_run, np.array(rels), k)


def ndcg_mean_and_std_portion(
    trajectories_of_all_runs: List[PredTrajs],
    portion: float,
    all_ids: List[str],
    all_group_labels: List[int],
    scoring: str,
    rels: List[int],
    k: int = 50,
) -> Tuple[float, float]:
    scores_per_run = []
    for trajectories in trajectories_of_all_runs:
        scores = evaluation_helper.get_scores(
            trajectories, all_ids, all_group_labels, scoring=scoring, portion=portion
        )
        scores_per_run.append(np.array(scores))
    return _ndcg_mean_std_over_runs(scores_per_run, np.array(rels), k)


# ----------------------------- Curve builders -----------------------------

def _compute_curve(
    param_values: Iterable,
    compute_mean_std: Callable[[object], Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Generic curve computation given ‘param_values -> (mean,std)’ callback."""
    means, stds = [], []
    for p in param_values:
        m, s = compute_mean_std(p)
        means.append(m)
        stds.append(s)
    return np.array(means), np.array(stds)


def ndcg_curve_initial_scoring(
    trajectories_of_all_runs: List[PredTrajs],
    all_ids: List[str],
    all_group_labels: List[int],
    k: int = 50,
    label: Optional[str] = None,
    color: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    include_mistakes: bool = False,
) -> Tuple[float, float, int]:
    """
    score^(k) with min-exc, k in {1..50}. Plots on the given ax (or current ax).
    Returns (max_ndcg, max_ndcg_std, argmax_index).
    """
    if ax is None:
        ax = plt.gca()

    rels = evaluation_helper.get_rels(all_group_labels, include_mistakes)
    max_min_length = 50
    ks = np.arange(1, max_min_length + 1, dtype=int)

    def compute_mean_std_for_k(min_len: int):
        return ndcg_mean_and_std_top_k(
            trajectories_of_all_runs, min_len, all_ids, all_group_labels, "min-exc", rels, k=k, include_mistakes=include_mistakes
        )

    curve_mean, curve_std = _compute_curve(ks, compute_mean_std_for_k)

    # print the classic checkpoints
    _print_keypoints(curve_mean, curve_std)

    # find max
    argmax = int(np.argmax(curve_mean))
    max_ndcg, max_ndcg_std = float(curve_mean[argmax]), float(curve_std[argmax])

    # plot
    _plot_curve(
        ax=ax,
        x=ks,
        mean=curve_mean,
        std=curve_std,
        label=label,
        color=color,
        mark_indices=[0, 1, 4, 9, 19, 49],  # show x/markers at 1,2,5,10,20,50
        #title=r"Results score$^{(k)}$ trajectory ranking",
        xlabel=r"$k$ ($k$-th largest per-detection score)",
        ylabel="NDCG@50",
    )
    return max_ndcg, max_ndcg_std, argmax+1


def ndcg_curve_topk(
    trajectories_of_all_runs: List[PredTrajs],
    all_ids: List[str],
    all_group_labels: List[int],
    scoring: str,
    k: int = 50,
    label: Optional[str] = None,
    color: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    max_min_length: int = 250,
    include_mistakes: bool = False,
) -> Tuple[float, float, int]:
    """score_{med/avg/wavg}^{(k)} with min_len in {1..max_min_length}."""
    if ax is None:
        ax = plt.gca()

    rels = evaluation_helper.get_rels(all_group_labels, include_mistakes=include_mistakes)
    ks = np.arange(1, max_min_length + 1, dtype=int)

    def compute_mean_std_for_k(min_len: int):
        return ndcg_mean_and_std_top_k(
            trajectories_of_all_runs, min_len, all_ids, all_group_labels, scoring, rels, k=k, include_mistakes=include_mistakes
        )

    curve_mean, curve_std = _compute_curve(ks, compute_mean_std_for_k)
    argmax = int(np.argmax(curve_mean))
    max_ndcg, max_ndcg_std = float(curve_mean[argmax]), float(curve_std[argmax])

    title_map = {
        "avg": r"Trajectory ranking score$_{avg}^{(k)}$",
        "med": r"Trajectory ranking score$_{med}^{(k)}$",
        "weighted-avg": r"Trajectory ranking score$_{wavg}^{(k)}$",
    }
    title = title_map.get(scoring, "NDCG Results")

    _plot_curve(
        ax=ax, x=ks, mean=curve_mean, std=curve_std,
        label=label, color=color, title=title, xlabel=r"$k$ ($k$-th largest per-detection score)", ylabel="NDCG@50"
    )
    return max_ndcg, max_ndcg_std, argmax+1


def ndcg_curve_percentil(
    trajectories_of_all_runs: List[PredTrajs],
    all_ids: List[str],
    all_group_labels: List[int],
    scoring: str,
    k: int = 50,
    label: Optional[str] = None,
    color: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[float, float, float]:
    """score_{med/avg/wavg}^{(p)} with p in [0.01, 0.99]."""
    if ax is None:
        ax = plt.gca()

    rels = evaluation_helper.get_rels(all_group_labels)
    portions = np.linspace(0.01, 1, 100)

    def compute_mean_std_for_p(pct: int):
        return ndcg_mean_and_std_portion(
            trajectories_of_all_runs, pct, all_ids, all_group_labels, scoring, rels, k=k
        )

    curve_mean, curve_std = _compute_curve(portions, compute_mean_std_for_p)
    argmax = int(np.argmax(curve_mean))
    max_ndcg, max_ndcg_std, best_portion = float(curve_mean[argmax]), float(curve_std[argmax]), float(portions[argmax])

    title_map = {
        "avg": r"Trajectory ranking score$_{avg}^{(p)}$",
        "med": r"Trajectory ranking score$_{med}^{(p)}$",
        "weighted-avg": r"Trajectory ranking score$_{wavg}^{(p)}$",
    }
    title = title_map.get(scoring, "NDCG Results")

    _plot_curve(
        ax=ax, x=portions, mean=curve_mean, std=curve_std,
        label=label, color=color, title=title,
        xlabel=r'Percentile $p$', ylabel="NDCG@50"
    )
        
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.05, 0.1, 0.2, 0.5, 1])
    ax.set_xticklabels(["99th", "95th", "90th", "80th", "50th", "0th"])

    return max_ndcg, max_ndcg_std, best_portion


def ndcg_curve_exp_weighted_avg(
    trajectories_of_all_runs: List[PredTrajs],
    all_ids: List[str],
    all_group_labels: List[int],
    k: int = 50,
    label: Optional[str] = None,
    color: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    include_mistakes: bool = False,
) -> Tuple[float, float, float]:
    """Full-trajectory, exp-weighted-avg; sweep weight_param a in [0.01, 1.06]."""
    if ax is None:
        ax = plt.gca()

    rels = evaluation_helper.get_rels(all_group_labels, include_mistakes=include_mistakes)
    weight_params = np.linspace(0.67, 1.06, 150)

    def compute_mean_std_for_a(a: float):
        return ndcg_mean_and_std_full_tr_scoring(
            trajectories_of_all_runs, all_ids, all_group_labels,
            scoring="exp-weighted-avg", rels=rels, k=k, weight_param=a, include_mistakes=include_mistakes
        )

    curve_mean, curve_std = _compute_curve(weight_params, compute_mean_std_for_a)
    if label == "ASYM1":
        ax.scatter([0.98], curve_mean[(np.abs(weight_params - 0.98)).argmin()], marker="x", s=70, color="black", zorder=5)
    argmax = int(np.argmax(curve_mean))
    max_ndcg, max_ndcg_std, best_param = float(curve_mean[argmax]), float(curve_std[argmax]), float(weight_params[argmax])

    _plot_curve(
        ax=ax, x=weight_params, mean=curve_mean, std=curve_std,
        label=label, color=color,
        #title=r"Results full trajectory ranking, exp-weighted-avg",
        xlabel=r"weight parameter $a$", ylabel="NDCG@50"
    )
    ax.set_xlim(np.min(weight_params), np.max(weight_params))
    return max_ndcg, max_ndcg_std, best_param
