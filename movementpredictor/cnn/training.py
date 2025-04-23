import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import copy
from typing import Tuple, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import logging

from movementpredictor.data import dataset
from movementpredictor.config import ModelConfig
from  movementpredictor.cnn import model_architectures, camera_angle

log = logging.getLogger(__name__)
config = ModelConfig()


def trainAndStoreCNN(path_data, path_model, architecture, output_prob, pixel_per_axis) -> Tuple[nn.Module, Dict[str, list[float]]]: 
  os.makedirs(path_model, exist_ok=True)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model_architectures.get_model(architecture=architecture, output_prob=output_prob)

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=2, verbose=True)
  history = dict(train=[], val=[])
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0

  model = model.train()
  train_losses = []
  val_losses = []
  no_improvement = 0

  train_ds = dataset.getTorchDataSet(os.path.join(path_data, "train"), pixel_per_axis)
  slope, intercept = camera_angle.calculate_camera_angle(train_ds, pixel_per_axis)
  val_ds = dataset.getTorchDataSet(os.path.join(path_data, "test"), pixel_per_axis, val_split=True)
  train = dataset.getTorchDataLoader(train_ds)
  val = dataset.getTorchDataLoader(val_ds)
    
  print("train size: ", len(train))
  print("val size: ", len(val))

  for epoch in range(100):

    for count, (input, target, _, _) in tqdm(enumerate(train)):
      
      optimizer.zero_grad()
      target = target.to(device)
      input = input.to(device)
      prediction = model(input)
      loss = model.loss(target, prediction, slope, intercept)
      loss.backward()
      optimizer.step()
      train_losses.append(loss.item())
      
      if (count+1) % 10000 == 0:  # check after 10000 batches
        model = model.eval()
        with torch.no_grad():
          for input, target, _, _ in val:
            target = target.to(device)
            input = input.to(device)
            prediction = model(input)
            loss = model.loss(target, prediction, slope, intercept)
            val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        print("iteration " + str(count), "- train_loss=" + str(train_loss) + " - val_loss=" + str(val_loss))
        if val_loss < best_loss:
          best_loss = val_loss
          best_model_wts = copy.deepcopy(model.state_dict())
          torch.save(best_model_wts, path_model + "/model_weights.pth")
          no_improvement = 0
        else: 
          print("no improvement")
          no_improvement += 1

        if no_improvement > 4: 
          break
        
        model = model.train()
        scheduler.step(val_loss)
        train_losses = []
        val_losses = []

    if no_improvement > 4:
      break

  model.load_state_dict(best_model_wts)
  model.eval()

  return model, history


def store_parameters(history, config: ModelConfig):
   timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

   paras = {
      "time": timestamp,
      "model_name": config.name_model,
      "model_architecture": config.model_architecture,
      "output_distribution": config.output_distribution,
      "training_data": os.path.basename(config.path_sae_data),
      "pixel_per_axis": config.pixel_per_axis,
    }
   
   with open(config.path_model + "/parameters.json", "w") as json_file:
      json.dump(paras, json_file, indent=4)
      


def plot_loss_curve(history, path):
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'], label='Val Loss')
    plt.xlabel('Iteration (in 10000 steps)')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(path, "loss_curve.png"))
    plt.show()
    plt.clf()


