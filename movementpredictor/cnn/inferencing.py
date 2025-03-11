import torch
from scipy.stats import multivariate_normal
from tqdm import tqdm
import numpy as np

from movementpredictor.data import dataset


def inference_with_stats(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples_with_stats = []

    with torch.no_grad():
        for i, (x, target, ts, id) in tqdm(enumerate(dataloader)):
            #if i == 10000:
             #   break
            x = torch.tensor(x).to(device)

            mu_batch, cov_batch = model(x)
            mu_batch, cov_batch = mu_batch.detach().cpu().numpy(), cov_batch.detach().cpu().numpy()

            for inp, mu, cov, pos, timestamp, obj_id in zip(x, mu_batch, cov_batch, target, ts, id):
                # for getting a measure of the likeliness of the target given the predicted normal distribution, the cdf is used
                #cov = cov + 1e-4 * np.eye(2)     # ensure pos. sem. definit
                cov = regularize_cov(cov)
                print(pos)
                print(mu)
                print(cov)
                print(get_bounding_box_info(inp))
                p = multivariate_normal.cdf(pos, mean=mu, cov=cov)
                prob = min(p, 1-p)

                stats = {
                    "input": get_bounding_box_info(inp),
                    "target": pos,
                    "prediction": {"mean": mu, "variance": cov, "probability_of_target": prob},
                    "timestamp": timestamp,
                    "obj_id": obj_id
                }
                samples_with_stats.append(stats)

    return samples_with_stats


def regularize_cov(cov, max_cond=6, min_var=1e-4):
    eigvals, eigvecs = np.linalg.eigh(cov)  

    cond_number = eigvals.max() / eigvals.min()  
    if cond_number > max_cond:
        #print(f"Regularizing covariance matrix. Original cond: {cond_number}")
        eigvals = np.maximum(eigvals, eigvals.max() / max_cond)  
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T  

    cov += min_var * np.eye(cov.shape[0])  

    return cov


def get_bounding_box_info(input):
    y_indices, x_indices = torch.where((input[-1] != 0) | (input[-2] != 0))  # bbox = 1

    x_min, x_max = x_indices.min().item(), x_indices.max().item()
    y_min, y_max = y_indices.min().item(), y_indices.max().item()
    
    return [[x_min, y_min], [x_max, y_max]]