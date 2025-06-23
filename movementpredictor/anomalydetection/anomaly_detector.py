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
from movementpredictor.anomalydetection import visualizer

from visionapi.sae_pb2 import SaeMessage
from visionlib import saedump

log = logging.getLogger(__name__)


def calculate_trajectory_anomaly_scores(samples_with_stats: List[inferencing.InferenceResult], a=0.94):
    trajectories = defaultdict(list)
    for sample in samples_with_stats:
        trajectories[sample.obj_id].append(sample.prediction.distance_of_target)

    trajectories_with_score = []
    for obj_id, distances in trajectories.items():
        if len(distances) < 10:
            continue
        sorted_measures = np.array(sorted(distances, reverse=True))
        score = 0
        sum_weights = 0
        for rank, measure in enumerate(sorted_measures):
            weight = a**rank
            score += measure*weight
            sum_weights += weight
        score = score/sum_weights
        trajectories_with_score.append([obj_id, score])
    
    return trajectories_with_score


def calculate_trajectory_threshold(samples_with_stats: List[inferencing.InferenceResult], percentage_p=None, 
                            num_anomalous_trajectories=None):
    """
    computes threshold so that 'percentage_p' percent of object IDs (trajectories) are considered normal.
    """
    if percentage_p is None and num_anomalous_trajectories is None:
        log.error("both, percentage_p and num_anomalous_trajectories are None. You have to specify one.")
        exit(1)
    elif percentage_p is not None and num_anomalous_trajectories is not None:
        log.warning("both, percentage_p and num_anomalous_trajectories are specified. You should specify only one. percentage_p will be used now.")

    total_obj_ids = set(sample.obj_id for sample in samples_with_stats)
    num_obj_ids_total = len(total_obj_ids)
    if num_anomalous_trajectories is not None and num_anomalous_trajectories > num_obj_ids_total:
        log.error("you want more anomalous trajectories than total trajectories are provided: total trajectories = " + 
                  str(num_obj_ids_total) + ", num_anomalous_trajectories = " + str(num_anomalous_trajectories))
        exit(1)

    # start caculating anomaly score for each trajectory
    trajectories_with_score = calculate_trajectory_anomaly_scores(samples_with_stats)

    # find threshold
    trajectories_with_score.sort(key=lambda x: x[1], reverse=True)
    num_obj_ids_target = int(np.ceil((100 - percentage_p) / 100 * num_obj_ids_total)) if percentage_p else num_anomalous_trajectories
    threshold = trajectories_with_score[num_obj_ids_target-1][1]

    print("Exp-weighted avg threshold: " + str(threshold))
    return threshold


def visualize_distances(samples_with_stats: List[inferencing.InferenceResult], path_plots):
    #threshold_dists, anomaly_obj_ids = calculate_threshold(samples_with_stats, percentage_p, num_anomalous_trajectories)
    dists = [sample.prediction.distance_of_target for sample in samples_with_stats]

    plt.hist(dists, bins=100, edgecolor='black')
    plt.title('distance distribution')
    plt.xlabel('dist')
    plt.ylabel('amount')
    #plt.axvline(x=threshold_dists, color='black', linestyle='dashed', label='threshold')

    os.makedirs(path_plots, exist_ok=True)
    path = os.path.join(path_plots, "distances.png")
    plt.savefig(path)
    plt.show()
    plt.clf()
    
    
def get_unlikely_trajectories(samples_with_stats: List[inferencing.InferenceResult], score_threshold) -> List[inferencing.InferenceResult]:
    anomaly_samples: List[inferencing.InferenceResult] = []
    trajectory_scores = calculate_trajectory_anomaly_scores(samples_with_stats)
    score_dict = {obj_id: score for obj_id, score in trajectory_scores}

    samples_by_trajectory = defaultdict(list)
    for sample in samples_with_stats:
        samples_by_trajectory[sample.obj_id].append(sample)

    selected_samples = []
    selected_trajectories = []
    for obj_id, samples in samples_by_trajectory.items():
        score = score_dict.get(obj_id, 0.0)
        
        if score >= score_threshold:
            selected_samples.extend(samples)
            selected_trajectories.append((obj_id, samples))

    all_distances = [s.prediction.distance_of_target for s in selected_samples]
    mean_dist = np.mean(all_distances)
    std_dist = np.std(all_distances)
    # 1-sigma border
    sigma_threshold = mean_dist + std_dist

    for obj_id, samples in selected_trajectories:
        over_sigma = [s for s in samples if s.prediction.distance_of_target > sigma_threshold]

        if len(over_sigma) >= 5:
            anomaly_samples.extend(over_sigma)
        else:
            # min 5 anomaly samples per anomaly trajectory
            top_5 = sorted(samples, key=lambda s: s.prediction.distance_of_target, reverse=True)[:5]
            anomaly_samples.extend(top_5)

    
    return anomaly_samples


def calculate_threshold(samples_with_stats: List[inferencing.InferenceResult], percentage_p=None, 
                            num_anomalous_trajectories=None, num_anomalous_frames_per_id=3):
    """
    computes threshold so that 'percentage_p' percent of object IDs (trajectories) are considered normal.
    """
    if percentage_p is None and num_anomalous_trajectories is None:
        log.error("both, percentage_p and num_anomalous_trajectories are None. You have to specify one.")
        exit(1)
    elif percentage_p is not None and num_anomalous_trajectories is not None:
        log.warning("both, percentage_p and num_anomalous_trajectories are specified. You should specify only one. percentage_p will be used now.")

    dist_obj_pairs = [(sample.prediction.distance_of_target, sample.obj_id) for sample in samples_with_stats]
    dist_obj_pairs.sort(reverse=True, key=lambda x: x[0])

    total_obj_ids = set(sample.obj_id for sample in samples_with_stats)
    num_obj_ids_total = len(total_obj_ids)
    if num_anomalous_trajectories is not None and num_anomalous_trajectories > num_obj_ids_total:
        log.error("you want more anomalous trajectories than total trajectories are provided: total trajectories = " + 
                  str(num_obj_ids_total) + ", num_anomalous_trajectories = " + str(num_anomalous_trajectories))
        exit(1)
        
    num_obj_ids_target = int(np.ceil((100 - percentage_p) / 100 * num_obj_ids_total)) if percentage_p else num_anomalous_trajectories
    obj_id_with_dists = defaultdict(list)
    anomaly_obj_ids = set()
    threshold_dists = None

    for dist, obj_id in dist_obj_pairs:
        obj_id_with_dists[obj_id].append(dist)
        if len(obj_id_with_dists[obj_id]) >= num_anomalous_frames_per_id:
            anomaly_obj_ids.add(obj_id)
        if len(anomaly_obj_ids) >= num_obj_ids_target:
            threshold_dists = dist
            break

    log.info("Distance-threshold: " + str(threshold_dists))
    return threshold_dists, anomaly_obj_ids


def calculate_and_visualize_threshold(samples_with_stats: List[inferencing.InferenceResult], path_plots, percentage_p=None, 
                                      num_anomalous_trajectories=None, num_anomalous_frames_per_id=3):
    
    threshold_dists, anomaly_obj_ids = calculate_threshold(samples_with_stats, percentage_p, num_anomalous_trajectories, num_anomalous_frames_per_id)
    dists = [sample.prediction.distance_of_target for sample in samples_with_stats]

    plt.hist(dists, bins=100, edgecolor='black')
    plt.title('distance distribution')
    plt.xlabel('dist')
    plt.ylabel('amount')
    plt.axvline(x=threshold_dists, color='black', linestyle='dashed', label='threshold')

    os.makedirs(path_plots, exist_ok=True)
    path = os.path.join(path_plots, "distances.png")
    plt.savefig(path)
    plt.show()
    plt.clf()

    return threshold_dists, anomaly_obj_ids



def plot_unlikely_samples(samples_with_stats, frame, test, threshold_dist, path_plots):
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

    if samples_with_stats[0].prediction.lambda_skew is not None:
        skews = [np.array(sample.prediction.lambda_skew) for sample in samples_with_stats]
        skews = [skews[i:i + batch_size] for i in range(0, len(skews), batch_size)]
    else:
        skews = [[None] * batch_size for _ in range(len(mus))]

    path = os.path.join(path_plots, "anomalies")
    os.makedirs(path, exist_ok=True)
    for i, (x, target, _, _) in tqdm(enumerate(test)):
        #if i == 1000:
        #   break
        for mu, cov, skew, inp, pos, dist in zip(mus[i], covs[i], skews[i], x, target, dists[i]):
            if dist > threshold_dist:
                count += 1
                visualizer.plot_input_target_output(frame, inp, pos, mu, cov, skew=skew)
                plt.title("Anomaly with distance " + str(dist))
                plt.savefig(os.path.join(path, "anomaly_" + str(count) + ".png"))
                plt.close()


def get_unlikely_samples(samples_with_stats: List[inferencing.InferenceResult], dist_thr, anomaly_obj_ids) -> List[inferencing.InferenceResult]:
    anomaly_samples: List[inferencing.InferenceResult] = []

    for sample in samples_with_stats:
        if sample.obj_id in anomaly_obj_ids and sample.prediction.distance_of_target >= dist_thr:
            anomaly_samples.append(sample)
    
    return anomaly_samples


def anomalies_with_video(anomalies: List[inferencing.InferenceResult], path_sae_dump, pixel_per_axis, path_plots):
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
        path = os.path.join(path_plots, "anomalies/" + str(dict_count))
        os.makedirs(path, exist_ok=True)
        dict_count += 1
        img_count = 0

        for ts, anomaly in video_dict[key][0]:
            # create plots
            frame_infos = [frame_info for frame_info in video_dict[key][1:] if frame_info[0].timestamp_utc_ms == ts]
            
            frame_info = frame_infos[0]
            frame_tensor = get_downsampled_tensor_img(frame_info[0], pixel_per_axis)

            skew = anomaly.prediction.lambda_skew
            mask_interest_np = create_mask_tensor(pixel_per_axis, [anomaly.input], scale=False).numpy()
            visualizer.make_plot(frame_tensor.numpy(), mask_interest_np, np.array(anomaly.target), np.array(anomaly.prediction.mean), 
                    np.array(anomaly.prediction.variance), dist=anomaly.prediction.distance_of_target, skew_lambda=np.array(skew) if skew is not None else None)
            
            plt.savefig(os.path.join(path, "a" + str(int(img_count)) + ".png"))
            img_count += 1
            plt.close()
        
        store_video(video_dict[key][1:], path)


def find_intervals_containing_timestamp(timestamp, intervals):
    return [interval for interval in intervals if interval[1] <= timestamp <= interval[2]]


def store_parameter(path_model, dist_thr, percentage_anomaly):
        with open(path_model + "/parameters.json", "r") as json_file:
            paras = json.load(json_file)
        
        paras["percentage_anomaly"] = percentage_anomaly
        paras["anomaly_threshold"] = dist_thr

        with open(path_model + "/parameters.json", "w") as json_file:
            json.dump(paras, json_file, indent=4)