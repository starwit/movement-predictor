import torch
from tqdm import tqdm
import numpy as np
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Optional


class PredictionStats(BaseModel):
    mean: List[float]
    variance: List[List[float]]
    lambda_skew: Optional[List[float]] = None
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
        for i, (x, target, ts, id) in tqdm(enumerate(dataloader), desc="inferencing movement predictor CNN"):
            #if i == 1000:
             #   break
            
            x = x.to(device)
            target = target.to(device)
            prediction = model(x)

            #if len(output) == 2:
             #   mu_batch, cov_batch = output
                #mu_batch, cov_batch = mu_batch.cpu(), cov_batch.cpu()
            #elif len(output) == 3:
             #   mu_batch, cov_batch, lambda_skew_batch = output
                #mu_batch, cov_batch, lambda_skew_batch = mu_batch.cpu(), cov_batch.cpu(), lambda_skew_batch.cpu()
            
            #target = target.cpu()

            ModelArchitecure = type(model)
            mahalanobis, _ = ModelArchitecure.mahalanobis_distance(target, prediction)
            mahalanobis = torch.sqrt(mahalanobis.squeeze())

            lambda_skew_batch = [None for _ in range(len(target))]
            if len(prediction) == 2:
                mu_batch, cov_batch = prediction
            elif len(prediction) == 3:
                mu_batch, cov_batch, lambda_skew_batch = prediction
                lambda_skew_batch = lambda_skew_batch.cpu().numpy()

            mu_batch, cov_batch, mahalanobis, target = mu_batch.cpu().numpy(), cov_batch.cpu().numpy(), mahalanobis.cpu().numpy(), target.cpu().numpy()

            for inp, mu, cov, lambda_skew, pos, timestamp, obj_id, dist in zip(x, mu_batch, cov_batch, lambda_skew_batch, target, ts, id, mahalanobis):
                stats = InferenceResult(
                    input=get_bounding_box_info(inp),
                    target=pos.tolist(),
                    prediction=PredictionStats(mean=mu.tolist(), variance=cov.tolist(), distance_of_target=dist),
                    timestamp=timestamp,
                    obj_id=obj_id
                )
                if lambda_skew is not None:
                    stats.prediction.lambda_skew = lambda_skew

                samples_with_stats.append(stats)
        
    return samples_with_stats


'''
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

    return cov'
'''


def get_bounding_box_info(input):
    y_indices, x_indices = torch.where((input[-1] != 0) | (input[-2] != 0))  # bbox = 1

    x_min, x_max = x_indices.min().item(), x_indices.max().item()
    y_min, y_max = y_indices.min().item(), y_indices.max().item()
    
    return [[x_min, y_min], [x_max, y_max]]
