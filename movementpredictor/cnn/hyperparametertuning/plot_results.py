import json
import optuna
from optuna.trial import FrozenTrial, TrialState
import os
import datetime
from movementpredictor.config import ModelConfig
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


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
            "lr": optuna.distributions.LogUniformDistribution(1e-5, 1e-2),
            #"scheduler_factor": optuna.distributions.FloatDistribution(0.1, 0.5),
            #"use_coord_channels": optuna.distributions.CategoricalDistribution([True, False]),
            #"max_loss_scale": optuna.distributions.FloatDistribution(0.1, 1.0)
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

print("\nTop 3 Trials based on Validation Loss:")
sorted_by_val = sorted(filtered_study.trials, key=lambda t: t.value)
for i, trial in enumerate(sorted_by_val[:3]):
    print(f"\nTrial {trial.number}:")
    print(f"  Validation Loss: {trial.value}")
    print(f"  Training Loss: {trial.user_attrs['train_loss_per_epoch'][-1]}")
    print(f"  Params: {trial.params}")


#optuna.visualization.plot_param_importances(study).show()
#optuna.visualization.plot_parallel_coordinate(filtered_study).show()
optuna.visualization.plot_slice(filtered_study).show()
#optuna.visualization.plot_optimization_history(study).show()


'''
data = []
for t in filtered_study.trials:
    if t.state == optuna.trial.TrialState.COMPLETE:
        data.append({
            "lr": t.params["lr"],
            "scheduler_factor": t.params["scheduler_factor"],
            "val_loss": t.value
        })

df = pd.DataFrame(data)
df["log_lr"] = np.log10(df["lr"])

# 2D-Heatmap erstellen
fig = px.scatter(
    df,
    x="log_lr",
    y="scheduler_factor",
    color="val_loss",
    color_continuous_scale="Viridis",
    labels={
        "log_lr": "log10(Learning Rate)",
        "scheduler_factor": "Scheduler Factor",
        "val_loss": "Validation Loss"
    },
    title="Validation Loss über Hyperparameter",
)

fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(coloraxis_colorbar=dict(title="Validation Loss"))
fig.show()


# 3D plot
params_lr = []
params_scheduler = []
final_val = []

for t in filtered_study.trials:
    if t.state == optuna.trial.TrialState.COMPLETE:
        params_lr.append(t.params["lr"])
        params_scheduler.append(t.params["scheduler_factor"])
        final_val.append(t.value)

params_lr = np.array(params_lr)
params_scheduler = np.array(params_scheduler)
final_val = np.array(final_val)

# 3D-Scatter-Plot
fig = go.Figure(data=[go.Scatter3d(
    x=params_lr,
    y=params_scheduler,
    z=final_val,
    mode='markers',
    marker=dict(
        size=5,
        color=final_val,    
        colorscale='Viridis',
        colorbar_title='Validation Loss',
        opacity=0.8
    )
)])

fig.update_layout(
    scene=dict(
        xaxis_title='Learning Rate',
        yaxis_title='Scheduler Factor',
        zaxis_title='Validation Loss',
        xaxis_type='log',  
    ),
    title='Validation Loss in Abhängigkeit von Hyperparametern',
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()
'''