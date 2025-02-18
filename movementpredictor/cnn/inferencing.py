import torch
from scipy.stats import multivariate_normal
from tqdm import tqdm
import numpy as np

from movementpredictor.data import dataset


def inference_with_stats(model, path_data: str, folder:str):
    # TODO: possiblity to also store targets and input masks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probs = []
    var_size = []
    mus, covs = [], []
    inputs, targets = [], []
    tss, ids = [], []

    with torch.no_grad():
        ds = dataset.merge_datasets(path_data, folder)
        test = dataset.getTorchDataLoader(ds, shuffle=False)
        print(len(test))

        for i, (x, target, ts, id) in tqdm(enumerate(test)):
            #if i == 10000:
             #   break
            x = torch.tensor(x).to(device)

            mu_batch, cov_batch = model(x)
            mu_batch, cov_batch = mu_batch.detach().cpu().numpy(), cov_batch.detach().cpu().numpy()
            mus.append(mu_batch)
            covs.append(cov_batch)
            targets.append(target)
            tss.append(ts)
            ids.append(id)
            inputs.append(get_bounding_box_info(x))

            for mu, cov, pos in zip(mu_batch, cov_batch, target):
                # for getting a measure of the likeliness of the target given the predicted normal distribution, the cdf is used
                #cov = cov + 1e-4 * np.eye(2)     # ensure pos. sem. definit
                cov = regularize_cov(cov)
                p = multivariate_normal.cdf(pos, mean=mu, cov=cov)
                prob = min(p, 1-p)

                probs.append(prob)
                variance = np.array([np.diag(v).sum() for v in cov]).sum()
                var_size.append(variance)

    return probs, var_size, mus, covs, inputs, targets, [tss, ids]


def regularize_cov(cov, max_cond=6, min_var=1e-4):
    eigvals, eigvecs = np.linalg.eigh(cov)  

    cond_number = eigvals.max() / eigvals.min()  
    if cond_number > max_cond:
        #print(f"Regularizing covariance matrix. Original cond: {cond_number}")
        eigvals = np.maximum(eigvals, eigvals.max() / max_cond)  
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T  

    cov += min_var * np.eye(cov.shape[0])  

    return cov


def get_bounding_box_info(batch):
    bboxs = []
    for i in range(batch.shape[0]):
        y_indices, x_indices = torch.where((batch[i][-1] != 0) | (batch[i][-2] != 0))  # bbox = 1

        x_min, x_max = x_indices.min().item(), x_indices.max().item()
        y_min, y_max = y_indices.min().item(), y_indices.max().item()
        
        bboxs.append([[x_min, y_min], [x_max, y_max]])

    return bboxs