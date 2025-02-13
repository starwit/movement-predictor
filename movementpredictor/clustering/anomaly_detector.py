import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from tqdm import tqdm
from movementpredictor.data import dataset
from movementpredictor.cnn import inferencing
import logging
from collections import Counter, defaultdict

log = logging.getLogger(__name__)


def plot_input_target_output(x, y, mu, sigma):
    frame_np = x[0].cpu().numpy()
    mask_others_np = x[1].cpu().numpy()
    mask_interest_np = x[2].cpu().numpy()
    target = y.cpu().numpy()

    make_plot(frame_np, mask_interest_np, target, mu, sigma, mask_others_np)


def make_plot(frame_np, mask_interest_np, target, mu, sigma, mask_others_np=None):
    plt.figure(figsize=(22, 7))

    plt.subplot(1, 3, 1)
    plt.title("input")
    plt.imshow(frame_np, cmap='gray', interpolation='nearest')
    if mask_others_np is not None:
        plt.imshow(mask_others_np, cmap='Reds', alpha=0.4, interpolation='nearest')
    plt.imshow(mask_interest_np, cmap='Blues', alpha=0.3, interpolation='nearest')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("target")
    frame_np = (frame_np * 255).astype(np.uint8)
    frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2RGB)
    cv2.circle(frame_rgb, [round(target[0]*frame_np.shape[-1]), round(target[1]*frame_np.shape[-2])], radius=2, color=(255, 0, 0), thickness=-1)
    plt.imshow(frame_rgb)
    if mask_others_np is not None:
        plt.imshow(mask_others_np, cmap='Reds', alpha=0.4, interpolation='nearest')
    plt.imshow(mask_interest_np, cmap='Blues', alpha=0.3, interpolation='nearest')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("prediction")
    frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2RGB)
    circle = [round(mu[0]*frame_np.shape[-1]), round(mu[1]*frame_np.shape[-2])]
    cv2.circle(frame_rgb, circle, radius=2, color=(255, 0, 0), thickness=-1)

    sigma = inferencing.regularize_cov(sigma)
    eigenvalues, eigenvectors = np.linalg.eigh(sigma*frame_np.shape[-1]*20)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    major_axis_length = 2 * np.sqrt(eigenvalues[0])  
    minor_axis_length = 2 * np.sqrt(eigenvalues[1]) 
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])) 
    
    cv2.ellipse(
        frame_rgb,
        center=circle,
        axes=(int(major_axis_length), int(minor_axis_length)),
        angle=angle,
        startAngle=0,
        endAngle=360,
        color=(255, 0, 0),
        thickness=1
    )

    plt.imshow(frame_rgb)
    if mask_others_np is not None:
        plt.imshow(mask_others_np, cmap='Reds', alpha=0.4, interpolation='nearest')
    plt.imshow(mask_interest_np, cmap='Blues', alpha=0.3, interpolation='nearest')
    plt.axis('off')


def visualValidation(model, path_data) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = dataset.getTorchDataSet(path_data, "clustering", 0)
    test0 = dataset.getTorchDataLoader(ds, train=False)

    with torch.no_grad():
        for count, (x, target, _, _) in enumerate(test0):
            if count >= 80: break
            model_data = torch.tensor(x).to(device)
            mu, sigma = model(model_data)
            #print(sigma)

            plot_input_target_output(x[0], target[0], mu[0].detach().cpu().numpy(), sigma[0].detach().cpu().numpy())
            plt.savefig("plots/outputAE" + str(count) + ".png")
            plt.close()


def output_distribution(probs, var_size, percentage_p=0.04, percentage_var=99.999):
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
    for i, (x, target, _, _) in tqdm(enumerate(test)):
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

   
def get_meaningful_unlikely_samples(probs, mus, covs, inputs, targets, prob_thr, ad_info):
    batch_size = len(mus[1])
    probs = [probs[i:i + batch_size] for i in range(0, len(probs), batch_size)]

    # 4 - bbox input; 2 - target position; 2 - output mean; 3 - cholesky of output cov 
    anomaly_mus = []
    anomaly_covs = []
    anomaly_inputs = []
    anomaly_targets = []
    anomaly_ts = []
    anomaly_id = []
    anomaly_probs = []

    for mu_batch, cov_batch, inp_batch, pos_batch, p_batch, ts_batch, id_batch in zip(mus, covs, inputs, targets, probs, ad_info[0], ad_info[1]):
        for mu, cov, inp, pos, p, ts, id in zip(mu_batch, cov_batch, inp_batch, pos_batch, p_batch, ts_batch, id_batch):
            if p < prob_thr:
                anomaly_mus.append(mu)
                anomaly_covs.append(cov)
                anomaly_inputs.append(inp)
                anomaly_targets.append(pos)
                anomaly_ts.append(ts)
                anomaly_id.append(id)
                anomaly_probs.append(p)
    
    # remove samples whose id only exists once
    id_counts = Counter(anomaly_id)
    ids_to_remove = {id for id, count in id_counts.items() if count == 1}
    indices_to_remove = [index for index, id in enumerate(anomaly_id) if id in ids_to_remove]

    anomaly_mus = [mu for index, mu in enumerate(anomaly_mus) if index not in indices_to_remove]
    anomaly_covs = [cov for index, cov in enumerate(anomaly_covs) if index not in indices_to_remove]
    anomaly_inputs = [inp for index, inp in enumerate(anomaly_inputs) if index not in indices_to_remove]
    anomaly_targets = [pos for index, pos in enumerate(anomaly_targets) if index not in indices_to_remove]
    anomaly_ts = [ts for index, ts in enumerate(anomaly_ts) if index not in indices_to_remove]
    anomaly_id = [id for index, id in enumerate(anomaly_id) if index not in indices_to_remove]
    anomaly_probs = [p for index, p in enumerate(anomaly_probs) if index not in indices_to_remove]

    return anomaly_inputs, anomaly_targets, anomaly_mus, anomaly_covs, anomaly_probs, anomaly_ts, anomaly_id