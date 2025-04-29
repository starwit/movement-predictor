import optuna
import os
import json
import numpy as np
from movementpredictor.cnn.training import train_model

from movementpredictor.config import ModelConfig

config = ModelConfig()
os.makedirs(config.path_model, exist_ok=True)


def objective(trial):
    #architecture = trial.suggest_categorical("architecture", ["SimpleCNN", "ResNet18", "MobileNet_v3"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    #scheduler_factor = trial.suggest_float("scheduler_factor", 0.1, 0.5)
    #coord_channels = trial.suggest_categorical("use_coord_channels", [True, False])

    model, history = train_model(config.path_store_data, config.path_model, config.model_architecture, config.output_distribution, config.pixel_per_axis,
                                lr=lr)#, scheduler_factor=scheduler_factor) #coord_channels=coord_channels)
    
    trial.set_user_attr("train_loss_per_epoch", history["train"])
    trial.set_user_attr("val_loss_per_epoch",   history["val"])
    best_loss = history["best_val_loss"]

    return best_loss


def save_study_callback(study, trial):
    results = []
    for t in study.trials:
        results.append({
            "trial": t.number,
            "params": t.params,
            "train_history": t.user_attrs.get("train_loss_per_epoch", []),
            "val_history":   t.user_attrs.get("val_loss_per_epoch", []),
            "final_val":     t.value
        })

    save_path = os.path.join(config.path_model, "optuna_full_history.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved intermediate results after Trial {trial.number} ✅")


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10, callbacks=[save_study_callback])