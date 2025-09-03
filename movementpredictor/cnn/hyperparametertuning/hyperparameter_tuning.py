import optuna
import os
import json
import numpy as np
import logging

from movementpredictor.cnn.training import train_model
from movementpredictor.config import ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

config = ModelConfig()
os.makedirs(config.path_model, exist_ok=True)


def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)

    model, history = train_model(config.path_store_data, config.path_model, config.model_architecture, config.output_distribution, config.pixel_per_axis,
                                lr=lr, save_model=False)
    
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

    log.info(f"Saved intermediate results after Trial {trial.number}")


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10, callbacks=[save_study_callback])