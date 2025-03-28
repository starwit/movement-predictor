import json
from typing import List
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from tqdm import tqdm
import logging
from collections import Counter, defaultdict
import math
import pybase64

from movementpredictor.data.datamanagement import get_downsampled_tensor_img
from movementpredictor.data.dataset import create_mask_tensor
from movementpredictor.anomalydetection.video_generation import store_video
from movementpredictor.cnn import inferencing

from visionapi.sae_pb2 import SaeMessage
from visionlib import saedump

log = logging.getLogger(__name__)


def plot_input_target_output(x, y, mu, sigma):
    frame_np = x[0].cpu().numpy()
    target = y.cpu().numpy()

    mask_others_np_sin = x[1].cpu().numpy()
    mask_others_np_cos = x[2].cpu().numpy()
    mask_others_np = np.zeros(frame_np.shape)
    mask_others_np[(mask_others_np_sin != 0) | (mask_others_np_cos != 0)] = 1

    mask_interest_np_sin = x[3].cpu().numpy()
    mask_interest_np_cos = x[4].cpu().numpy()
    mask_interest_np = np.zeros(frame_np.shape)
    mask_interest_np[(mask_interest_np_sin != 0) | (mask_interest_np_cos != 0)] = 1
        
    # calculate angle
    sin = np.max(mask_interest_np_sin) if np.max(mask_interest_np_sin) > 0 else np.min(mask_interest_np_sin)
    cos = np.max(mask_interest_np_cos) if np.max(mask_interest_np_cos) > 0 else np.min(mask_interest_np_cos)
    angle_rad = math.atan2(sin, cos)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    angle_deg = round(angle_deg/2)

    make_plot(frame_np, mask_interest_np, target, mu, sigma, mask_others_np, angle=angle_deg)


def make_plot(frame_np, mask_interest_np, target, mu, sigma, mask_others_np=None, angle=None, dist=None):
    plt.figure(figsize=(22, 7))

    plt.subplot(1, 3, 1)
    plt.title("input") if angle is None else plt.title("input, orientation angle: " + str(angle))
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
    plt.title("prediction") if dist is None else plt.title("prediction, distance=" + str(dist))
    frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2RGB)
    circle = [round(mu[0]*frame_np.shape[-1]), round(mu[1]*frame_np.shape[-2])]
    cv2.circle(frame_rgb, circle, radius=2, color=(255, 0, 0), thickness=-1)

    #sigma = inferencing.regularize_cov(sigma)
    eigenvalues, eigenvectors = np.linalg.eigh(sigma*frame_np.shape[-1]*0.1)  # add factor for better visualization
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


def visualValidation(model, dataloader, num_plots=100) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for count, (x, target, _, _) in enumerate(dataloader):
            if count >= num_plots: break
            model_data = torch.tensor(x).to(device)
            mu, sigma = model(model_data)

            plot_input_target_output(x[0], target[0], mu[0].detach().cpu().numpy(), sigma[0].detach().cpu().numpy())
            plt.savefig("plots/outputAE" + str(count) + ".png")
            plt.close()


def calculate_and_visualize_threshold(samples_with_stats: List[inferencing.InferenceResult], percentage_p=99.99, path=None):
    '''computes the thresholds for finding an anomaly based on the variance based distance to the mean'''

    dists = [sample.prediction.distance_of_target for sample in samples_with_stats]
    threshold_dists = np.percentile(dists, percentage_p)
    log.info("Distance-threshold: " + str(threshold_dists))

    plt.hist(dists, bins=100, edgecolor='black')
    plt.title('distance distribution')
    plt.xlabel('dist')
    plt.ylabel('amount')
    plt.axvline(x=threshold_dists, color='black', linestyle='dashed', label='threshold')
    plt.savefig("plots/distances.png" if path is None else path)
    plt.show()
    plt.clf()

    return threshold_dists

'''
    var_size = [np.diag(sample["prediction"]["variance"]).sum() for sample in samples_with_stats]
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

    return threshold_dists, threshold_vars''
'''


def store_parameter(path_model, dist_thr, percentage_anomaly):
    with open(path_model + "/parameters.json", "r") as json_file:
        paras = json.load(json_file)
    
    paras["percentage_anomaly"] = percentage_anomaly
    paras["anomaly_threshold"] = dist_thr

    with open(path_model + "/parameters.json", "w") as json_file:
        json.dump(paras, json_file, indent=4)


def plot_unlikely_samples(test, threshold_dist, samples_with_stats: List[inferencing.InferenceResult]):
    count = 0
    batch_size = test.batch_size

    dists = [sample.prediction.distance_of_target for sample in samples_with_stats]
    #var_size = [np.diag(sample["prediction"]["variance"]).sum() for sample in samples_with_stats]
    mus = [np.array(sample.prediction.mean) for sample in samples_with_stats]
    covs = [np.array(sample.prediction.variance) for sample in samples_with_stats]

    dists = [dists[i:i + batch_size] for i in range(0, len(dists), batch_size)]
    #var_size = [var_size[i:i + batch_size] for i in range(0, len(var_size), batch_size)]
    mus = [mus[i:i + batch_size] for i in range(0, len(mus), batch_size)]
    covs = [covs[i:i + batch_size] for i in range(0, len(covs), batch_size)]

    os.makedirs("plots/anomalies", exist_ok=True)
    for i, (x, target, _, _) in tqdm(enumerate(test)):
        for mu, cov, inp, pos, dist in zip(mus[i], covs[i], x, target, dists[i]):
            if dist > threshold_dist:
                count += 1
                plot_input_target_output(inp, pos, mu, cov)
                plt.title("Anomaly with distance " + str(dist))
                plt.savefig("plots/anomalies/anomaly_" + str(count) + ".png")
                plt.close()
            #elif var_s > threshold_vars:
             #   count += 1
              #  plot_input_target_output(inp, pos, mu, cov)
               # plt.title("anomaly due to large variance: var size = " + str(var_s))
               # plt.savefig("plots/anomalies/anomaly_" + str(count) + ".png")
                #plt.close()


def find_intervals_containing_timestamp(timestamp, intervals):
    return [interval for interval in intervals if interval[1] <= timestamp <= interval[2]]


def anomalies_with_video(anomalies: List[inferencing.InferenceResult], path_sae_dump, dim_x, dim_y):
    anomaly_dict = defaultdict(list)
    anomaly_ts_int = [int(anomaly.timestamp) for anomaly in anomalies]

    for anomaly, ts in zip(anomalies, anomaly_ts_int):
        anomaly_dict[anomaly.obj_id].append([ts, anomaly])
    
    video_dict = defaultdict(list)
    for key in anomaly_dict.keys():
        timestamps = [value[0] for value in anomaly_dict[key]]
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        start = min_ts - 3000
        end = max_ts + 3000
        video_dict[(key, start, end)] = [anomaly_dict[key]]
    
    with open(path_sae_dump, 'r') as input_file:
        messages = saedump.message_splitter(input_file)

        start_message = next(messages)
        saedump.DumpMeta.model_validate_json(start_message)

        for message in tqdm(messages, desc="collecting frames"):
            event = saedump.Event.model_validate_json(message)
            proto_bytes = pybase64.standard_b64decode(event.data_b64)

            proto = SaeMessage()
            proto.ParseFromString(proto_bytes)
            frame_ts = proto.frame.timestamp_utc_ms

            fitting_keys = find_intervals_containing_timestamp(frame_ts, video_dict.keys())

            for key in fitting_keys:
                frame_info = [proto.frame, None]
                for detection in proto.detections:
                    if key[0] == str(detection.object_id.hex()):
                        frame_info[1] = detection.bounding_box
                        break
                video_dict[key].append(frame_info)

    dict_count = 0
    for key in video_dict:
        path = "plots/anomalies/" + str(dict_count) + "/"
        os.makedirs(path, exist_ok=True)
        dict_count += 1
        img_count = 0

        for ts, anomaly in video_dict[key][0]:
            # create plots
            frame_infos = [frame_info for frame_info in video_dict[key][1:] if frame_info[0].timestamp_utc_ms == ts]
            
            frame_info = frame_infos[0]
            frame_tensor = get_downsampled_tensor_img(frame_info[0], dim_x, dim_y)

            mask_interest_np = create_mask_tensor(dim_x, dim_y, [anomaly.input], scale=False).numpy()
            make_plot(frame_tensor.numpy(), mask_interest_np, np.array(anomaly.target), np.array(anomaly.prediction.mean), 
                      np.array(anomaly.prediction.variance), dist=anomaly.prediction.distance_of_target)
            
            plt.savefig(path + "a" + str(int(img_count)) + ".png")
            img_count += 1
            plt.close()
        
        store_video(video_dict[key][1:], path)


def get_meaningful_unlikely_samples(samples_with_stats: List[inferencing.InferenceResult], dist_thr) -> List[inferencing.InferenceResult]:
    # 4 - bbox input; 2 - target position; 2 - output mean; 3 - cholesky of output cov 
    anomaly_samples = get_unlikely_samples(samples_with_stats, dist_thr)
    
    # remove samples whose id only exists once
    anomaly_ids = [sample.obj_id for sample in anomaly_samples]
    id_counts = Counter(anomaly_ids)
    ids_to_remove = {id for id, count in id_counts.items() if count == 1}
    indices_to_remove = [index for index, id in enumerate(anomaly_ids) if id in ids_to_remove]

    anomalies = [sample for index, sample in enumerate(anomaly_samples) if index not in indices_to_remove]

    return anomalies


def get_unlikely_samples(samples_with_stats: List[inferencing.InferenceResult], dist_thr) -> List[inferencing.InferenceResult]:
    anomaly_samples: List[inferencing.InferenceResult] = []

    for sample in samples_with_stats:
        if sample.prediction.distance_of_target > dist_thr:
            anomaly_samples.append(sample)
    
    return anomaly_samples