import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from typing import Tuple, Dict
from tqdm import tqdm
import copy
from scipy.stats import multivariate_normal
#from pytorch_tabular.models import mixture_density, MDNConfig

def trainAndStoreAE(train: DataLoader, val: DataLoader, path: str) -> Tuple[nn.Module, Dict[str, list[float]]]: 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = MixtureDensityNetwork()
  model.to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
  criterion = mdn_nll_loss
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
  history = dict(train=[], val=[])
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0

  model = model.train()
  train_losses = []
  val_losses = []
  print("train size: ", len(train))
  print("val size: ", len(val))
  no_improvement = 0

  for epoch in range(100):
    for count, (input, target) in tqdm(enumerate(train)):
      optimizer.zero_grad()
      target = target.to(device)
      input = input.to(device)
      mu, sigma, pi = model(input)
      loss = criterion(target, mu, sigma, pi)
      loss.backward()
      optimizer.step()
      train_losses.append(loss.item())
      
      if count % int(10000/input.shape[0]) == 0:  # check after 10000 images
        model = model.eval()
        with torch.no_grad():
          for input, target in val:
            target = target.to(device)
            input = input.to(device)
            mu, sigma, pi = model(input)
            loss = criterion(target, mu, sigma, pi)
            val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        print("iteration " + str(count), "- train_loss=" + str(train_loss) + " - val_loss=" + str(val_loss))
        if val_loss < best_loss:
          best_loss = val_loss
          best_model_wts = model.state_dict()
          torch.save(best_model_wts, path)
          no_improvement = 0
        else: 
          print("no improvement")
          no_improvement += 1

        if no_improvement > 17: 
          break
        
        model = model.train()
        scheduler.step(val_loss)
        train_losses = []
        val_losses = []

    if no_improvement > 15:
      break

  model.load_state_dict(best_model_wts)
  model.eval()

  return model, history


class MixtureDensityNetwork(nn.Module):
    def __init__(self, num_gaussians=3):
        super(MixtureDensityNetwork, self).__init__()
        self.num_gaussians = num_gaussians

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)   
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  
        self.poolavg = nn.AvgPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.poolmax = nn.MaxPool2d(2, 2)                                 
        
        # Fully connected layers
        self.fc = nn.Linear(256 * 5 * 5, 100)
        self.fc_mu = nn.Linear(100, self.num_gaussians * 2)  # Output: predicted means for "num_gaussian" Gaussians 
        self.fc_cov = nn.Linear(100, self.num_gaussians * 3)  # Output: covariance parameters for "num_gaussian" Gaussians
        self.fc_pi = nn.Linear(100, self.num_gaussians)       # Output: mixing coefficients for "num_gaussian" Gaussians
        
    def forward(self, x):
        # Encoder (simple CNN)
        x = nn.ReLU()(self.conv1(x))                # shape 32x40x40
        x = nn.ReLU()(self.conv2(x))                # shape 64x20x20
        x = self.poolavg(nn.ReLU()(self.conv3(x)))  # shape 128x10x10
        x = nn.ReLU()(self.conv4(x))                # shape 256x10x10
        x = self.poolmax(nn.ReLU()(self.conv5(x)))  # shape 256x5x5
        
        # Flatten the output from the convolutional layers
        x = x.view(-1, 256 * 5 * 5)
        
        x = nn.ReLU()(self.fc(x))
        # Output: means, covariance parameters, mixing coefficients
        mu = self.fc_mu(x).view(-1, self.num_gaussians, 2)  # Reshape to (batch_size, num_gaussians, 2)
        cov_params = self.fc_cov(x).view(-1, self.num_gaussians, 3)  # Reshape to (batch_size, num_gaussians, 3)
        pi = nn.Softmax(dim=1)(self.fc_pi(x))  # Reshape to (batch_size, num_gaussians) with softmax
        pi = torch.clamp(pi, min=0.1)  # at minimum 0.1
        pi = pi / pi.sum(dim=1, keepdim=True) # normaization: sum to 1
        
        # Construct covariance matrices (positive semi-definite)
        L = torch.zeros(mu.size(0), self.num_gaussians, 2, 2).to(x.device)
        L[:, :, 0, 0] = torch.nn.functional.softplus(cov_params[:, :, 0]) + 1e-5  # Ensure positive value
        L[:, :, 1, 0] = cov_params[:, :, 1] # Off-diagonal
        L[:, :, 0, 1] = cov_params[:, :, 1] # Off-diagonal
        L[:, :, 1, 1] = torch.nn.functional.softplus(cov_params[:, :, 2]) + 1e-5 # Ensure positive value
        
        # Construct covariance matrices
        L_reshaped = L.view(-1, 2, 2)  # Shape: (batch_size * num_gaussians, 2, 2)
        sigma = torch.bmm(L_reshaped, L_reshaped.transpose(1, 2))
        sigma = sigma.view(mu.size(0), self.num_gaussians, 2, 2)

        return mu, sigma, pi



def mdn_nll_loss(y_true, mu, sigma, pi, lambda_var=0.05, lambda_separation=0.02):
    """
    Negative Log-Likelihood (NLL) mit angepassten Mindestwerten für Mischgewichte.
    
    :param y_true: Wahre Zielwerte (batch_size, 2)
    :param mu: Mittelwerte der Gaussians (batch_size, K, 2)
    :param sigma: Kovarianzmatrizen der Gaussians (batch_size, K, 2, 2)
    :param pi: normalisierte Logits der Mischgewichte (batch_size, K)
    :param lambda_var: Gewicht der Varianzbegrenzung
    :param lambda_separation: Gewicht des Separationsloss
    :return: Verlustwert
    """
    n_gaussians = mu.size(1)
    batch_size = y_true.size(0)
    epsilon = 1e-4  # Stabilisierungskonstante

    # Negative Log-Likelihood (NLL)
    log_likelihoods = []
    for k in range(n_gaussians):
        mu_k = mu[:, k, :]
        sigma_k = sigma[:, k, :, :]

        # Stabilisierung der Kovarianzmatrizen
        sigma_stable = sigma_k + epsilon * torch.eye(sigma_k.size(-1), device=sigma_k.device)
        L = torch.linalg.cholesky(sigma_stable)
        L_inv = torch.cholesky_inverse(L)

        diff = y_true - mu_k
        mahalanobis = torch.sum(torch.sum(diff.unsqueeze(-1) * torch.bmm(L_inv, diff.unsqueeze(2)), dim=1), dim=1)
        log_det_sigma = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=1)

        log_pi_k = torch.log(pi[:, k])
        log_likelihood = -0.5 * (mahalanobis + lambda_var * log_det_sigma + 2 * torch.log(torch.tensor(2 * torch.pi, device=y_true.device)))
        log_likelihoods.append(log_likelihood + log_pi_k)

    log_likelihoods = torch.stack(log_likelihoods, dim=1)
    log_prob = torch.logsumexp(log_likelihoods, dim=1)
    nll_loss = -torch.mean(log_prob)

    # Separationsloss
    trace_sigma = torch.sum(torch.diagonal(sigma, dim1=-2, dim2=-1), dim=-1)
    separations = []
    for i in range(n_gaussians):
        for j in range(i + 1, n_gaussians):
            diff_mu = mu[:, i, :] - mu[:, j, :]
            dist_mu = torch.sum(diff_mu**2, dim=-1)
            avg_sigma = 0.5 * (trace_sigma[:, i] + trace_sigma[:, j])
            separations.append(torch.exp(-dist_mu / (2 * avg_sigma + epsilon)))
    separations = torch.stack(separations, dim=1) if separations else torch.zeros(1, device=y_true.device)
    separation_loss = lambda_separation * torch.mean(separations)

    # Gesamtverlust
    total_loss = nll_loss + separation_loss
    return total_loss

