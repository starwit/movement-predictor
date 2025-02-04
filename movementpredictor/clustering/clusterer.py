import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import numpy as np
import cv2
from scipy.stats import multivariate_normal
from tqdm import tqdm
import hdbscan
from sklearn.decomposition import PCA
from movementpredictor.data import dataset
from movementpredictor.cnn import inferencing


def plot_input_target_output(x, y, mu, sigma):
    frame_np = x[0].cpu().numpy()
    mask_np = x[1].cpu().numpy()
    target = y.cpu().numpy()

    plt.figure(figsize=(22, 7))

    plt.subplot(1, 3, 1)
    plt.title("input")
    plt.imshow(frame_np, cmap='gray', interpolation='nearest')
    plt.imshow(mask_np, cmap='Reds', alpha=0.5, interpolation='nearest')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("target")
    frame_np = (frame_np * 255).astype(np.uint8)
    frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2RGB)
    cv2.circle(frame_rgb, [round(target[0]), round(target[1])], radius=4, color=(255, 0, 0), thickness=-1)
    plt.imshow(frame_rgb)
    plt.imshow(mask_np, cmap='Reds', alpha=0.5, interpolation='nearest')
    plt.axis('off')


    plt.subplot(1, 3, 3)
    plt.title("prediction")
    frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2RGB)
    cv2.circle(frame_rgb, [round(mu[0]), round(mu[1])], radius=2, color=(255, 0, 0), thickness=-1)

    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    major_axis_length = 2 * np.sqrt(eigenvalues[0])  
    minor_axis_length = 2 * np.sqrt(eigenvalues[1]) 
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])) 
    
    cv2.ellipse(
        frame_rgb,
        center=(round(mu[0]), round(mu[1])),
        axes=(int(major_axis_length), int(minor_axis_length)),
        angle=angle,
        startAngle=0,
        endAngle=360,
        color=(255, 0, 0),
        thickness=1
    )

    plt.imshow(frame_rgb)
    plt.imshow(mask_np, cmap='Reds', alpha=0.5, interpolation='nearest')
    plt.axis('off')


def visualValidation(model, path_data) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = dataset.getTorchDataSet(path_data, "clustering", 0)
    test0 = dataset.getTorchDataLoader(ds, train=False)

    with torch.no_grad():
        for count, (x, target) in enumerate(test0):
            if count >= 100: break
            model_data = torch.tensor(x).to(device)
            mu, sigma = model(model_data)
            #print(sigma)

            plot_input_target_output(x[0], target[0], mu[0].detach().cpu().numpy(), sigma[0].detach().cpu().numpy())
            plt.savefig("plots/outputAE" + str(count) + ".png")
            plt.close()


def output_distribution(probs, var_size, percentage_p=0.01, percentage_var=99.99999):
    '''computes the thresholds for finding an anomaly based on the probability density and variance'''

    threshold_probs = np.percentile(probs, percentage_p)
    print(threshold_probs)

    plt.hist(probs, bins=100, edgecolor='black')
    plt.title('probability distribution')
    plt.xlabel('prob')
    plt.ylabel('amount')
    plt.axvline(x=threshold_probs, color='black', linestyle='dashed', label='threshold')
    plt.savefig("plots/probs.png")
    plt.show()
    plt.clf()

    threshold_vars = np.percentile(var_size, percentage_var)
    print(threshold_vars)

    plt.hist(var_size, bins=50, edgecolor='black')
    plt.title('cov-size distribution')
    plt.xlabel('cov-size')
    plt.ylabel('amount')
    plt.axvline(x=threshold_vars, color='black', linestyle='dashed', label='threshold')
    plt.savefig("plots/cov-size.png")
    plt.show()
    plt.clf()

    return threshold_probs, threshold_vars


def plot_unlikely_samples(path_data, threshold_probs, threshold_vars, probs, var_size, mus, covs):
    count = 0
    ds = dataset.merge_datasets(path_data, "clustering")
    test = dataset.getTorchDataLoader(ds, train=False)

    batch_size = test.batch_size
    probs = [probs[i:i + batch_size] for i in range(0, len(probs), batch_size)]
    var_size = [var_size[i:i + batch_size] for i in range(0, len(var_size), batch_size)]

    os.makedirs("plots/anomalies", exist_ok=True)
    for i, (x, target) in tqdm(enumerate(test)):
        #if i >= 5000:
         #   break
        for mu, cov, inp, pos, p, var_s in zip(mus[i], covs[i], x, target, probs[i], var_size[i]):
            if p < threshold_probs:
                count += 1
                plot_input_target_output(inp, pos, mu, cov)
                plt.title("anomaly due to unlikeliness: prob. = " + str(p))
                plt.savefig("plots/anomalies/anomaly_" + str(count) + ".png")
                plt.close()
            elif var_s > threshold_vars:
                count += 1
                plot_input_target_output(inp, pos, mu, cov)
                plt.title("anomaly due to large variance: var size = " + str(var_s))
                plt.savefig("plots/anomalies/anomaly_" + str(count) + ".png")
                plt.close()

    return threshold_probs, threshold_vars


def output_distribution_pointMap(model, test:DataLoader, path, background):
    '''plots all the unlikely targets in a point map'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    directory = os.path.dirname(path)

    probs = []
    var_size = []
    mus, covs, pis = [], [], []
    with torch.no_grad():
        for x, target in tqdm(test):
            x = torch.tensor(x).to(device)
            mu_batch, cov_batch, pi_batch = model(x)
            mu_batch, cov_batch, pi_batch = mu_batch.detach().cpu().numpy(), cov_batch.detach().cpu().numpy(), pi_batch.detach().cpu().numpy()
            mus.append(mu_batch)
            covs.append(cov_batch)
            pis.append(pi_batch)
            for mu, cov, pi, pos in zip(mu_batch, cov_batch, pi_batch, target):
                # for getting a measure of the likeliness of the target given the predicted normal distribution, the cdf is used
                cov = cov + 1e-4 * np.eye(2)[np.newaxis, :, :]      # ensure pos. sem. definit
                prob = 0
                for m, v, weight in zip(mu, cov, pi):
                    p = multivariate_normal.cdf(pos, mean=m, cov=v)
                    prob += min(p, 1-p)*weight

                probs.append(prob)
                variance = np.array([np.diag(v).sum() for v in cov]).sum()
                var_size.append(variance)

    threshold_probs = np.percentile(probs, 0.02)
    print(threshold_probs)

    plt.hist(probs, bins=100, edgecolor='black')
    plt.title('probability distribution')
    plt.xlabel('prob')
    plt.ylabel('amount')
    plt.axvline(x=threshold_probs, color='black', linestyle='dashed', label='threshold')
    plt.savefig(directory + "/plots/probs.png")
    plt.show()
    plt.clf()

    threshold_vars = np.percentile(var_size, 99.99)
    print(threshold_vars)

    plt.hist(var_size, bins=50, edgecolor='black')
    plt.title('cov-size distribution')
    plt.xlabel('cov-size')
    plt.ylabel('amount')
    plt.axvline(x=threshold_vars, color='black', linestyle='dashed', label='threshold')
    plt.savefig(directory + "/plots/cov-size.png")
    plt.show()
    plt.clf()

    batch_size = test.batch_size
    probs = [probs[i:i + batch_size] for i in range(0, len(probs), batch_size)]
    var_size = [var_size[i:i + batch_size] for i in range(0, len(var_size), batch_size)]

    input_pos_prob = [[], []]
    target_pos_prob = [[], []]
    input_pos_var = [[], []]
    target_pos_var = [[], []]

    for i, (x, target) in tqdm(enumerate(test)):
        for mu, cov, pi, inp, pos, p, var_s in zip(mus[i], covs[i], pis[i], x, target, probs[i], var_size[i]):
            if p < threshold_probs:
                # TODO: extract input position from input image
                x_inp, y_inp = extract_coordinates_from_image(inp)
                if x_inp is not None:
                    input_pos_prob[0].append(x_inp)
                    input_pos_prob[1].append(y_inp)
                    target_pos_prob[0].append(pos[0] * background.shape[1])
                    target_pos_prob[1].append(pos[1] * background.shape[0])
            if var_s > threshold_vars:
                x_inp, y_inp = extract_coordinates_from_image(inp)
                if x_inp is not None:
                    input_pos_var[0].append(x_inp)
                    input_pos_var[1].append(y_inp)
                    target_pos_var[0].append(pos[0] * background.shape[1])
                    target_pos_var[1].append(pos[1] * background.shape[0])

    target_img = background.copy()
    plt.imshow(target_img)
    plt.scatter(input_pos_prob[0], input_pos_prob[1], c="b", s=10)
    plt.scatter(input_pos_var[0], input_pos_var[1], c="y", s=10)
    plt.scatter(target_pos_prob[0], target_pos_prob[1], c="r", s=10)
    plt.scatter(target_pos_var[0], target_pos_var[1], c="g", s=10)
    plt.savefig(directory + "/plots/unlikely_points.png")
    plt.clf()

    return (input_pos_prob, target_pos_prob, threshold_probs), (input_pos_var, target_pos_var, threshold_vars)


def extract_coordinates_from_image(inp: torch.tensor):
    red_mask = (inp[0, :, :] > 0.5) & (inp[1, :, :] < 0.3) & (inp[2, :, :] < 0.3)
    y_indices, x_indices = torch.where(red_mask)

    # calculate center 
    if len(x_indices) > 0 and len(y_indices) > 0:
        center_x = x_indices.float().mean().item()
        center_y = y_indices.float().mean().item()
        return center_x, center_y
    else:
        print("Did not find red cross")
        return None, None


def apply_clustering(prob_stats, var_stats, path):
    (in_prob, tar_prob), (in_var, tar_var) = prob_stats, var_stats
    input = torch.cat([torch.tensor(in_prob).reshape((-1, 2)), torch.tensor(in_var).reshape((-1, 2))], dim=0)
    target = torch.cat([torch.tensor(tar_prob).reshape((-1, 2)), torch.tensor(tar_var).reshape((-1, 2))], dim=0)
    data = torch.cat([input, target], dim=1).numpy()

    hdbscan_cluster = hdbscan.HDBSCAN(min_cluster_size=3)
    clusters = hdbscan_cluster.fit_predict(data)

    # visualization
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters, cmap="viridis")
    plt.colorbar(label='Cluster')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('Clusters in 2dim space')
    plt.savefig(os.path.dirname(path) + "/plots/Clusters_PCA.png")
    plt.clf()

    return clusters

