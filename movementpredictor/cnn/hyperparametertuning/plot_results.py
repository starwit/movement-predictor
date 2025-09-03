import json
import logging
import optuna
from optuna.trial import FrozenTrial, TrialState
import os
import datetime
from movementpredictor.config import ModelConfig
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

config = ModelConfig()
path_json = os.path.join(config.path_model, "optuna_full_history.json")

with open(path_json, "r") as f:
    results = json.load(f)

trials = []
for i, entry in enumerate(results):
    now = datetime.datetime.now()
    trial = FrozenTrial(
        number=entry["trial"],
        state=TrialState.COMPLETE,
        value=entry["final_val"],
        params=entry["params"],
        distributions={
            "lr": optuna.distributions.LogUniformDistribution(1e-6, 1e-2),
        },
        user_attrs={
            "train_loss_per_epoch": entry["train_history"],
            "val_loss_per_epoch": entry["val_history"],
        },
        system_attrs={},
        intermediate_values={},
        trial_id=i,
        datetime_start=now,
        datetime_complete=now,
    )
    trials.append(trial)

study = optuna.study.create_study(direction="minimize")
study.add_trials(trials)
outlier_threshold = np.percentile([t.value for t in study.trials], 100)  # 90% Quantil
filtered_trials = [t for t in study.trials if t.value <= outlier_threshold]
filtered_study = optuna.create_study(direction="minimize")
filtered_study.add_trials(filtered_trials)

log.info("\nTop 3 Trials based on Validation Loss:")
sorted_by_val = sorted(filtered_study.trials, key=lambda t: t.value)
for i, trial in enumerate(sorted_by_val[:3]):
    log.info(f"\nTrial {trial.number}:")
    log.info(f"  Validation Loss: {trial.value}")
    log.info(f"  Training Loss: {trial.user_attrs['train_loss_per_epoch'][-1]}")
    log.info(f"  Params: {trial.params}")


optuna.visualization.plot_slice(filtered_study).show()