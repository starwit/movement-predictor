import torch
from tqdm import tqdm
import numpy as np
from pydantic import BaseModel, ConfigDict
from typing import List, Dict


class PredictionStats(BaseModel):
    mean: List[float]
    variance: List[List[float]]
    distance_of_target: float


class InferenceResult(BaseModel):
    input: List[List[int]]
    target: List[float]
    prediction: PredictionStats
    timestamp: str
    obj_id: str


def inference_with_stats(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> List[InferenceResult]:
    """
    Perform inferencing on the model. Inferencing results and distance calculations (necessary to perform anomaly detection later) 
    are returned for each sample as an object of type InferenceResult

    Args: 
        model: pytorch model 
        dataloader: pytorch dataloader 
    
    Returns:
        list[InferenceResult]: a list with one entry (InferenceResult) for each sample in dataloader
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples_with_stats = []

    with torch.no_grad():
        for i, (x, target, ts, id) in tqdm(enumerate(dataloader)):
            if i == 10000:
                break
            x = x.clone().detach().to(device)

            mu_batch, cov_batch = model(x)
            mu_batch, cov_batch = mu_batch.detach().cpu().numpy(), cov_batch.detach().cpu().numpy()
            target = target.detach().cpu().numpy()

            for inp, mu, cov, pos, timestamp, obj_id in zip(x, mu_batch, cov_batch, target, ts, id):
                #cov = regularize_cov(cov)

                sigma_stable = cov + 1e-6 * np.eye(cov.shape[0])
                sigma_inv = np.linalg.inv(sigma_stable)
                diff = (pos - mu).reshape(-1, 1) 
                dist = np.matmul(np.matmul(diff.T, sigma_inv), diff).squeeze()
                dist = np.sqrt(dist)

                stats = InferenceResult(
                    input=get_bounding_box_info(inp),
                    target=pos.tolist(),
                    prediction=PredictionStats(mean=mu.tolist(), variance=cov.tolist(), distance_of_target=dist),
                    timestamp=timestamp,
                    obj_id=obj_id
                )
                samples_with_stats.append(stats)

    return samples_with_stats


def regularize_cov(cov, max_cond=8, min_achsis=0.01):
    #eigenvalues, eigenvectors = np.linalg.eigh(cov)
    #eigenvalues = np.maximum(eigenvalues, min_achsis)
    #cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    eigvals, eigvecs = np.linalg.eigh(cov)  

    cond_number = eigvals.max() / eigvals.min()  
    if cond_number > max_cond:
        #print(f"Regularizing covariance matrix. Original cond: {cond_number}")
        eigvals = np.maximum(eigvals, eigvals.max() / max_cond)  
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T  

    return cov


def get_bounding_box_info(input):
    y_indices, x_indices = torch.where((input[-1] != 0) | (input[-2] != 0))  # bbox = 1

    x_min, x_max = x_indices.min().item(), x_indices.max().item()
    y_min, y_max = y_indices.min().item(), y_indices.max().item()
    
    return [[x_min, y_min], [x_max, y_max]]