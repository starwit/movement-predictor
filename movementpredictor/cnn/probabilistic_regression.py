import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import copy
from typing import Tuple, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
#from torchvision.models import efficientnet_b0
from movementpredictor.data import dataset
from movementpredictor.config import ModelConfig
import json
from datetime import datetime


def trainAndStoreCNN(path_data, path_model) -> Tuple[nn.Module, Dict[str, list[float]]]: 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = CNN()
  #model = EfficientNetRegressionWithUncertainty()
  model.to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  criterion = nll_loss
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=4, verbose=True)
  history = dict(train=[], val=[])
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0

  model = model.train()
  train_losses = []
  val_losses = []
  no_improvement = 0

  train_ds, val_ds = dataset.merge_and_split_datasets(path_data)
  train = dataset.getTorchDataLoader(train_ds)
  val = dataset.getTorchDataLoader(val_ds)
    
  print("train size: ", len(train))
  print("val size: ", len(val))

  for epoch in range(5):

    for count, (input, target, _, _) in tqdm(enumerate(train)):
      
      optimizer.zero_grad()
      target = target.to(device)
      input = input.to(device)
      mu, sigma = model(input)
      loss = criterion(target, mu, sigma)
      loss.backward()
      optimizer.step()
      train_losses.append(loss.item())
      
      if (count+1) % 10000 == 0:  # check after 10000 batches
        model = model.eval()
        with torch.no_grad():
          for input, target, _, _ in val:
            target = target.to(device)
            input = input.to(device)
            mu, sigma = model(input)
            loss = criterion(target, mu, sigma)
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

        if no_improvement > 10: 
          break
        
        model = model.train()
        scheduler.step(val_loss)
        train_losses = []
        val_losses = []

    if no_improvement > 10:
      break

  model.load_state_dict(best_model_wts)
  model.eval()

  return model, history


def store_parameters(history, config: ModelConfig):
   timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

   paras = {
      "time": timestamp,
      "model_name": "CNN",
      "training_data": os.path.basename(config.path_sae_data),
      "dim_x": config.dim_x,
      "dim_y": config.dim_y
    }
   
   with open(config.path_model + "/parameters.json", "w") as json_file:
      json.dump(paras, json_file, indent=4)


class advancedCNN(nn.Module):
    def __init__(self):
        super(advancedCNN, self).__init__()

        # Convolutional layers with dilation instead of stride
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=4, dilation=4)

        # Residual blocks for better gradient flow
        self.res_block1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(32, 128)
        )

        self.res_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(32, 128)
        )

        # Downsampling with pooling
        self.pool1 = nn.AdaptiveAvgPool2d((30, 30))
        self.pool2 = nn.AdaptiveAvgPool2d((15, 15))

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 15 * 15, 512)
        self.fc2 = nn.Linear(512, 100)

        # Outputs
        self.fc_mu = nn.Linear(100, 2)  # Mean
        self.fc_cov = nn.Linear(100, 3)  # Covariance matrix parameters

    def forward(self, x):
        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Residual block 1
        res = x
        x = self.res_block1(x)
        x += res
        x = F.relu(x)

        # Apply first pooling layer
        x = self.pool1(x)

        # Residual block 2
        res = x
        x = self.res_block2(x)
        x += res
        x = F.relu(x)

        # Downsample to a fixed size
        x = self.pool2(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Mean output
        mu = self.fc_mu(x)

        # Covariance matrix parameters
        cov_params = self.fc_cov(x)

        # Construct covariance matrix (positive semi-definite)
        L = torch.zeros(mu.size(0), 2, 2).to(x.device)
        L[:, 0, 0] = F.softplus(cov_params[:, 0])
        L[:, 1, 0] = cov_params[:, 1]
        L[:, 0, 1] = cov_params[:, 1]
        L[:, 1, 1] = F.softplus(cov_params[:, 2])

        sigma = torch.bmm(L, L.transpose(1, 2))

        return mu, sigma


class CNN(nn.Module):
    """
    Convolutional Neural Network (pytorch model) to make a prediction on a vehicles position a certain time ahead.
    """

    def __init__(self):
        super(CNN, self).__init__()
        # Encoder part (for simplicity, let's use simple conv layers)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1)   
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  
        self.poolavg = nn.AvgPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.poolmax = nn.MaxPool2d(2, 2)                                      
        
        # Fully connected layers
        self.fc = nn.Linear(256 * 15 * 15, 100)
        self.fc_mu = nn.Linear(100, 2)  # Output: predicted mean (μ) 2 dimensions
        self.fc_cov = nn.Linear(100, 3)  # Output: covariance matrix (sigma): Cholesky decomposition (3 parameters for a 2x2 symmetric matrix)
    
    def forward(self, x):
        # Encoder (simple CNN)
        x = nn.ReLU()(self.conv1(x))                # shape 32x60x60
        x = self.poolavg(nn.ReLU()(self.conv2(x)))  # shape 64x30x30
        x = nn.ReLU()(self.conv3(x))                # shape 128x30x30
        x = nn.ReLU()(self.conv4(x))                # shape 256x30x30
        x = self.poolmax(nn.ReLU()(self.conv5(x)))  # shape 256x15x15
        
        # Flatten the output from the convolutional layers
        x = x.view(-1, 256 * 15 * 15)
        
        x = nn.ReLU()(self.fc(x))
        # Output: mean (μ) and covariance matrix (sigma)
        mu = self.fc_mu(x)
        # Predict Cholesky decomposition parameters for covariance (L)
        cov_params = self.fc_cov(x)
        
        # Construct covariance matrix (positive semi-definite)
        L = torch.zeros(mu.size(0), 2, 2).to(x.device)
        L[:, 0, 0] = torch.nn.functional.softplus(cov_params[:, 0])  # Ensure positive value
        L[:, 1, 0] = cov_params[:, 1] # Off-diagonal
        L[:, 0, 1] = cov_params[:, 1] # Off-diagonal
        L[:, 1, 1] = torch.nn.functional.softplus(cov_params[:, 2])  # Ensure positive value
        
        # Construct covariance matrix
        sigma = torch.bmm(L, L.transpose(1, 2))

        return mu, sigma


def nll_loss(y_true, mu, sigma):
    """
    Negative Log-Likelihood (NLL) Loss with regularization
    
    Args:
        y_true (torch.Tensor): Wahrer Wert, Form (batch_size, 2).
        mu (torch.Tensor): Erwartungswert (mean), Form (batch_size, 2).
        sigma (torch.Tensor): Kovarianzmatrix, Form (batch_size, 2, 2).
        
    Returns:
        torch.Tensor: Durchschnittlicher NLL-Wert über den Batch.
    """
    # Error term: (y_true - mu)
    error = (y_true - mu).unsqueeze(2)  # Shape (batch_size, 2, 1)
    
    epsilon = 1e-4  
    sigma_stable = sigma + epsilon * torch.eye(sigma.size(-1)).to(sigma.device) 
    
    eigenvalues = torch.linalg.eigvalsh(sigma_stable)  # Nur reelle Eigenwerte
    cond_number = torch.max(eigenvalues) / torch.min(eigenvalues + 1e-5)  # no zero-division
    
    #log_det = torch.logdet(sigma_stable)
    sign, log_det = torch.linalg.slogdet(sigma_stable)
    log_det = sign * log_det

    #selected_channels = input[:, 2:4, :, :]
    #non_zero_pixels = (selected_channels != 0).any(dim=1)  
    #pixel_counts = non_zero_pixels.sum(dim=(1, 2))

    sigma_inv = torch.inverse(sigma_stable)
    mahalanobis = torch.bmm(torch.bmm(error.transpose(1, 2), sigma_inv), error)
    
    # NLL Loss: Mahalanobis-Distanz + log(det(sigma)) + Regularisierung
    loss = mahalanobis.squeeze() + 0.01 * log_det + 0.0002 * cond_number 
    #loss =  mahalanobis.squeeze() + 0.01 * log_det + 0.001 * cond_number

    return loss.mean()
    


def plot_loss_curve(history):
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'], label='Val Loss')
    plt.xlabel('Iteration (in 10000 steps)')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig("plots/loss_curve.png")
    plt.show()
    plt.clf()

'''
class EfficientNetRegressionWithUncertainty(nn.Module):
   
  def __init__(self):
    super(EfficientNetRegressionWithUncertainty, self).__init__()

    self.model_base = efficientnet_b0(pretrained=True)
    self.model_base.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

    num_features = self.model_base.classifier[1].in_features  # input features for last layer
    self.model_base.classifier = nn.Sequential(
      nn.Linear(num_features, 128),  
      nn.ReLU() 
    )
    self.fc_mu = nn.Linear(128, 2)  # Output: predicted mean (μ) 2 dimensions
    self.fc_cov = nn.Linear(128, 3)  # Output: covariance matrix (sigma): Cholesky decomposition (3 parameters for a 2x2 symmetric matrix)

  def forward(self, x):
    x = self.model_base(x)

    # Output: mean (μ) and covariance matrix (sigma)
    mu = self.fc_mu(x)
    # Predict Cholesky decomposition parameters for covariance (L)
    cov_params = self.fc_cov(x)
    
    # Construct covariance matrix (positive semi-definite)
    L = torch.zeros(mu.size(0), 2, 2).to(x.device)
    L[:, 0, 0] = torch.nn.functional.softplus(cov_params[:, 0])  # Ensure positive value
    L[:, 1, 0] = cov_params[:, 1] # Off-diagonal
    L[:, 0, 1] = cov_params[:, 1] # Off-diagonal
    L[:, 1, 1] = torch.nn.functional.softplus(cov_params[:, 2])  # Ensure positive value
    
    # Construct covariance matrix (σ = L * L^T)
    sigma = torch.bmm(L, L.transpose(1, 2))

    return mu, sigma
'''
