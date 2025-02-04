import torch
from scipy.stats import multivariate_normal
from tqdm import tqdm
import numpy as np

from movementpredictor.data import dataset


def inference_with_prob_calculation(model, path_data: str, folder:str):
    # TODO: possiblity to also store targets and input masks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probs = []
    var_size = []
    mus, covs = [], []

    with torch.no_grad():
        ds = dataset.merge_datasets(path_data, folder)
        test = dataset.getTorchDataLoader(ds, train=False)
        print(len(test))

        for i, (x, target) in tqdm(enumerate(test)):
            #if i > 5000:
             #   break
            x = torch.tensor(x).to(device)

            mu_batch, cov_batch = model(x)
            mu_batch, cov_batch = mu_batch.detach().cpu().numpy(), cov_batch.detach().cpu().numpy()
            mus.append(mu_batch)
            covs.append(cov_batch)

            for mu, cov, pos in zip(mu_batch, cov_batch, target):
                # for getting a measure of the likeliness of the target given the predicted normal distribution, the cdf is used
                cov = cov + 1e-4 * np.eye(2)     # ensure pos. sem. definit
                p = multivariate_normal.cdf(pos, mean=mu, cov=cov)
                prob = min(p, 1-p)

                probs.append(prob)
                variance = np.array([np.diag(v).sum() for v in cov]).sum()
                var_size.append(variance)

    return probs, var_size, mus, covs


def inference_with_prob_calculation__(model, path_data: str, folder: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probs = []
    var_size = []
    mus, covs = [], []

    with torch.no_grad():
        
        for i in range(4):
            ds = dataset.getTorchDataSet(path_data, folder, i)
            test = dataset.getTorchDataLoader(ds, train=False)
            print(len(test))

            for i, (x, target) in tqdm(enumerate(test)):
                # Falls x bereits Tensor ist, brauchen wir keine Umwandlung
                x = x.to(device)

                # Berechnungen in einem Schritt, nur einmal `detach().cpu().numpy()`
                mu_batch, cov_batch = model(x)
                mu_batch_cpu, cov_batch_cpu = mu_batch.detach().cpu().numpy(), cov_batch.detach().cpu().numpy()

                # Anstatt für jedes Element die Schleife, könnte ein NumPy Vectorized Approach helfen
                mus.append(mu_batch_cpu)
                covs.append(cov_batch_cpu)

                # Für alle Elemente auf einmal berechnen, ohne Looping
                for mu, cov, pos in zip(mu_batch_cpu, cov_batch_cpu, target):
                    cov = cov + 1e-4 * np.eye(2)  # ensure pos. sem. definit
                    p = multivariate_normal.cdf(pos, mean=mu, cov=cov)
                    prob = min(p, 1 - p)

                    probs.append(prob)
                    variance = np.sum([np.diag(v).sum() for v in cov])
                    var_size.append(variance)

        return probs, var_size, mus, covs
    

def inference_with_prob_calculation_(model, path_data: str, folder: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probs = []
    var_size = []
    mus, covs = [], []

    with torch.no_grad():
        for i in range(4):
            ds = dataset.getTorchDataSet(path_data, folder, i)
            test = dataset.getTorchDataLoader(ds, train=False)
            print(len(test))

            for j, (x_batch, target_batch) in tqdm(enumerate(test)):
                if j > 100:
                    break
                x_batch = x_batch.to(device)

                mu_batch, cov_batch = model(x_batch)
                
                mu_batch_cpu = mu_batch.detach().cpu().numpy()
                cov_batch_cpu = cov_batch.detach().cpu().numpy()

                mus.append(mu_batch_cpu)
                covs.append(cov_batch_cpu)

                cov_batch_cpu += np.eye(2) * 1e-4

                p_batch = np.array([
                    multivariate_normal.cdf(pos, mean=mu, cov=cov)
                    for pos, mu, cov in zip(target_batch, mu_batch_cpu, cov_batch_cpu)
                ])
                probs.extend(np.minimum(p_batch, 1 - p_batch))

                var_batch = np.trace(cov_batch_cpu, axis1=-2, axis2=-1)
                var_size.extend(var_batch)


    return probs, var_size, mus, covs
