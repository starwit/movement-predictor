from movementpredictor.cnn import training
from movementpredictor.config import ModelConfig

import os
import json
import logging

log = logging.getLogger(__name__)
config = ModelConfig()



def main():
    
    path_json = os.path.join(config.path_model, "optuna_full_history.json")

    if os.path.exists(path_json):
        with open(path_json, "r") as f:
            results = json.load(f)
        best_trial = min(results, key=lambda x: x["final_val"])
        best_lr = best_trial["params"]["lr"]
        lr = float(f"{best_lr:.1e}")

        print("start training with lr=" + str(lr))
        model, history_paras = training.train_model(config.path_store_data, config.path_model, config.model_architecture, config.output_distribution, config.pixel_per_axis, lr=lr)
    else:
        model, history_paras = training.train_model(config.path_store_data, config.path_model, config.model_architecture, config.output_distribution, config.pixel_per_axis) 
    
    training.store_parameters(history_paras, config)

    training.plot_loss_curve(history_paras, config.path_plots)
    model.eval()
    
if __name__ == "__main__":
    main()