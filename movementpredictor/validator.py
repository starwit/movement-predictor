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


def make_output_img(img, data, color, weight):
    cv2.circle(
        img, 
        center=(round(data[0] * img.shape[1]), round(data[1] * img.shape[0])),
        radius=1,
        color=(int(color[0] * weight), int(color[1] * weight), int(color[2] * weight)),
        thickness=-1
    )
    return img

def plot_input_target_output(x, y, mu, sigma, pi, background):
    plt.figure(figsize=(22, 7))

    plt.subplot(1, 3, 1)
    plt.imshow(np.array(x.permute(1, 2, 0), np.uint8))
    plt.title("input")

    plt.subplot(1, 3, 2)
    target_img = background.copy()
    target_img = make_output_img(target_img, y.numpy(), color=(0, 255, 0), weight=1.0)
    plt.imshow(target_img)
    plt.title("target")

    plt.subplot(1, 3, 3)
    img = background.copy()
    for m, s, weight in zip(mu, sigma, pi):  # Iterate through Gaussians
        eigenvalues, eigenvectors = np.linalg.eigh(s * 80)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        major_axis_length = 2 * np.sqrt(eigenvalues[0])  
        minor_axis_length = 2 * np.sqrt(eigenvalues[1]) 
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])) 
        
        color = (255, 0, 0)
        #img = make_output_img(img, m, color=color, weight=weight)
        img = make_output_img(img, m, color=color, weight=1)
        print(m)
        print(weight)
        cv2.ellipse(
            img,
            center=(round(m[0] * img.shape[1]), round(m[1] * img.shape[0])),
            axes=(int(major_axis_length), int(minor_axis_length)),
            angle=angle,
            startAngle=0,
            endAngle=360,
            #color=(int(color[0] * weight), int(color[1] * weight), int(color[2] * weight)),
            color=(int(color[0]), int(color[1]), int(color[2])),
            thickness=1
        )
    print("-----")

    plt.imshow(img)


def visualValidation(model, test, path, background) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    directory = os.path.dirname(path)

    with torch.no_grad():
        for count, (x, target) in enumerate(test):
            if count >= 30: break
            model_data = torch.tensor(x).to(device)
            mu, sigma, pi = model(model_data)

            plot_input_target_output(x[0], target[0], mu[0].detach().cpu().numpy(), sigma[0].detach().cpu().numpy(),
                                     pi[0].detach().cpu().numpy(), background)
            plt.savefig(directory + "/plots/outputAE" + str(count) + ".png")
            plt.close()


def output_distribution(model, test:DataLoader, path, background):
    '''computes the thresholds for finding an anomaly based on the probability density and variance'''
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

    threshold_probs = np.percentile(probs, 0.005)
    print(threshold_probs)

    plt.hist(probs, bins=100, edgecolor='black')
    plt.title('probability distribution')
    plt.xlabel('prob')
    plt.ylabel('amount')
    plt.axvline(x=threshold_probs, color='black', linestyle='dashed', label='threshold')
    plt.savefig(directory + "/plots/probs.png")
    plt.show()
    plt.clf()

    threshold_vars = np.percentile(var_size, 99.999)
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

    count = 0
    for i, (x, target) in tqdm(enumerate(test)):
        for mu, cov, pi, inp, pos, p, var_s in zip(mus[i], covs[i], pis[i], x, target, probs[i], var_size[i]):
            if p < threshold_probs:
                count += 1
                plot_input_target_output(inp, pos, mu, cov, pi, background)
                plt.title("anomaly due to unlikeliness: prob. = " + str(p))
                plt.savefig(directory + "/plots/anomalies/anomaly_" + str(count) + ".png")
                plt.close()
            elif var_s > threshold_vars:
                count += 1
                plot_input_target_output(inp, pos, mu, cov, pi, background)
                plt.title("anomaly due to large variance: var size = " + str(var_s))
                plt.savefig(directory + "/plots/anomalies/anomaly_" + str(count) + ".png")
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

