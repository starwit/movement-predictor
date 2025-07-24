from typing import Dict
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2
import logging
import os
from collections import defaultdict
import pickle
import math
from movementpredictor.data.datamanagement import TrackedObjectPosition


log = logging.getLogger(__name__)


def getTorchDataSet(path_store, pixel_per_axis=120, val_split=False, ids_of_interest=None):
    all_data = []

    if os.path.isfile(path_store) and path_store.endswith(".pkl"):
        pkl_files = [path_store]
    else:
        pkl_files = [os.path.join(path_store, f) for f in os.listdir(path_store) if f.endswith(".pkl")]

    for file in pkl_files:
        with open(file, "rb") as f:
            data = pickle.load(f)  
            all_data.append(data)  

    raw_dataset = [item for data_list in all_data for item in data_list]
    if ids_of_interest is not None:
        raw_dataset = [item for item in raw_dataset if item[-2] in ids_of_interest]

    torch_dataset = CNNData(raw_dataset, pixel_per_axis)

    if val_split:
        torch.manual_seed(0)
        total_size = len(torch_dataset)
        val_size = int(min(total_size, 50000))
        _, val_ds = random_split(torch_dataset, [total_size - val_size, val_size])
        return val_ds

    return torch_dataset


def getTorchDataLoader(dataset, shuffle=True):
    batch_size = 16
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def makeTorchDataLoader(tracks: Dict[str, list[TrackedObjectPosition]], class_of_interest=2, time_diff_prediction=2, frame_rate=10, pixel_per_axis=120) -> DataLoader:
    """
    Creates a pytorch DataLoader to make the data progressible for pytorch deep learning models.
    Based on the tracking information the model's input (boundingboxes, movement_angles) 
    and the target (vehicles position "time_diff_prediction" seconds after input scene) are created and put into one dataset.
    
    Args:
        tracks: dict with trajectories (key: object id, value: list of this object's tracks)
        frame_rate: frame rate of video data
    
    Retruns: 
        DataLoader: pytorch Dataloader to make easy use of the dataset in batches

    """
    raw_dataset = make_input_target_pairs(tracks, class_of_interest, time_diff_prediction, frame_rate)
    if len(raw_dataset) == 0:
        return None
    torch_dataset = CNNData(raw_dataset, pixel_per_axis)
    torch_dataloader = getTorchDataLoader(torch_dataset, shuffle=False)
    return torch_dataloader


def store_data(tracks: dict, class_of_interest: int, path_store: str, time_diff_prediction: float, folder="test", frame_rate=10, name_dump=None):
    dataset = make_input_target_pairs(tracks, class_of_interest, time_diff_prediction, frame_rate)

    path = os.path.join(path_store, folder)
    os.makedirs(path, exist_ok=True)

    # basic file name
    base_filename = "dataset" if name_dump is None else name_dump
    filename = f"{base_filename}.pkl"

    full_path = os.path.join(path, filename)

    with open(full_path, "wb") as f:
        pickle.dump(dataset, f)


def make_input_target_pairs(tracks: dict, class_of_interest: int, time_diff_prediction: float, frame_rate: float = 10) -> Dataset:
    time_interval_in_millisec = 1100*time_diff_prediction
    prediction_step = 1000*time_diff_prediction
    input_target_pairs = []

    estimated_frames = round((time_interval_in_millisec / 1000) * frame_rate)

    for trajectory in tqdm(tracks.values(), desc="creating dataset - calculating target positions"):
        if len(trajectory) == 0:
            continue

        objects_of_interest = [input_tr.class_id == class_of_interest for input_tr in trajectory]
        if sum(objects_of_interest) / len(objects_of_interest) > 0.8:      # only examine vehicles that are at least 80% classified as a the class_id of interest -> high probability of actually being of this class

            for i, input_tr in enumerate(trajectory):
                if i + estimated_frames >= len(trajectory):
                    break

                if not input_tr.clear_detection:
                    continue

                ts_slice = trajectory[i: i + estimated_frames]
                ts_slice_smooth = [t for t in ts_slice if t.clear_detection]

                if len(ts_slice_smooth) < 0.8*estimated_frames:
                    continue

                duration = ts_slice_smooth[-1].capture_ts - input_tr.capture_ts

                # make sure the object is detected in at least 80% of the frames
                if duration*0.8 <=  time_interval_in_millisec:
                    input_target_pairs.append([input_tr, get_position_after_time(ts_slice_smooth, prediction_step)])


    timestamp_dict = defaultdict(list)
    
    for _, trajectory in tracks.items():
        for track in trajectory:
            timestamp_dict[track.capture_ts].append(track)

    for i, (inp, tar) in tqdm(enumerate(input_target_pairs), desc="creating dataset - collecting all bboxs"):
        other_vehicles = timestamp_dict[inp.capture_ts]
        bboxs = [vehicle.bbox for vehicle in other_vehicles]
        angles = [vehicle.movement_angle for vehicle in other_vehicles]

        bboxs_other = np.array(bboxs, dtype=np.float32)
        angles_other = np.array(angles, dtype=np.float32)
        input_bbox = np.array(inp.bbox, dtype=np.float32)
        target_pos = np.array(tar, dtype=np.float32)
        frame_ts = np.str_(inp.capture_ts)
        obj_id = np.str_(inp.uuid)
        angle = np.float32(inp.movement_angle)

        input_target_pairs[i] = [bboxs_other, angles_other, input_bbox, target_pos, frame_ts, obj_id, angle]
    
    return input_target_pairs


def get_position_after_time(ts, target_time):

    times = [track.capture_ts for track in ts]
    xs = [track.center[0] for track in ts]
    ys = [track.center[1] for track in ts]
    
    target_time_stamp = ts[0].capture_ts + target_time
    x_interp = np.interp(target_time_stamp, times, xs)
    y_interp = np.interp(target_time_stamp, times, ys)

    return [x_interp, y_interp]


def plotDataSamples(dataloader: DataLoader, amount: int, path: str, frame: torch.tensor):

    for count, (sample_batch, target_batch, _, _) in enumerate(dataloader):
        if count >= amount: break

        sample, target = sample_batch[0], target_batch[0]
        frame_np = frame.numpy()

        mask_others_np_sin = sample[0].cpu().numpy()
        mask_others_np_cos = sample[1].cpu().numpy()
        mask_others_np = np.zeros(frame_np.shape)
        mask_others_np[(mask_others_np_sin != 0) | (mask_others_np_cos != 0)] = 1

        mask_interest_np_sin = sample[-2].cpu().numpy()
        mask_interest_np_cos = sample[-1].cpu().numpy()
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

        target = target.cpu().numpy()

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title("input, orientation angle : " + str(angle_deg))
        plt.imshow(frame_np, cmap='gray', interpolation='nearest')
        plt.imshow(mask_others_np, cmap='Reds', alpha=0.4, interpolation='nearest')
        plt.imshow(mask_interest_np, cmap='Blues', alpha=0.3, interpolation='nearest')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("target")
        frame_np = (frame_np * 255).astype(np.uint8)
        frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2RGB)
        cv2.circle(frame_rgb, [round(target[0]*frame_np.shape[-1]), round(target[1]*frame_np.shape[-2])], radius=2, color=(255, 0, 0), thickness=-1)
        plt.imshow(frame_rgb)
        plt.imshow(mask_others_np, cmap='Reds', alpha=0.4, interpolation='nearest')
        plt.imshow(mask_interest_np, cmap='Blues', alpha=0.3, interpolation='nearest')
        plt.axis('off')

        parent_path = os.path.dirname(path)
        plt.savefig(os.path.join(parent_path, "exampleInput" + str(count) + ".png"))
        plt.close()

    
def plotMasksOnly(dataloader, amount: int, path: str):
    for count, (sample_batch, target_batch, _, _) in enumerate(dataloader):
        if count >= amount:
            break

        sample = sample_batch[0]

        m_others_sin     = sample[0].cpu().numpy()
        m_others_cos     = sample[1].cpu().numpy()
        m_interest_sin   = sample[-2].cpu().numpy()
        m_interest_cos   = sample[-1].cpu().numpy()

        fig, axes = plt.subplots(2, 2, figsize=(6, 6))

        axes[0, 0].imshow(m_others_sin, cmap='gray', interpolation='nearest', vmin=-1, vmax=1)
        axes[0, 0].set_title(r'Context channel$_\sin$')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(m_others_cos, cmap='gray', interpolation='nearest', vmin=-1, vmax=1)
        axes[0, 1].set_title(r'Context channel$_\cos$')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(m_interest_sin, cmap='gray', interpolation='nearest', vmin=-1, vmax=1)
        axes[1, 0].set_title(r'Target Vehicle channel$_\sin$')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(m_interest_cos, cmap='gray', interpolation='nearest', vmin=-1, vmax=1)
        axes[1, 1].set_title(r'Target Vehicle channel$_\cos$')
        axes[1, 1].axis('off')

        plt.tight_layout()

        # speichern
        parent = os.path.dirname(path)
        out_fn = os.path.join(parent, f"maskOnly_{count}.png")
        fig.savefig(out_fn, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def bbox_mask(pixel, bbox, scale):
    [x_min, y_min], [x_max, y_max] = bbox

    if scale:
        x_min_idx = int(x_min * pixel)
        x_max_idx = int(x_max * pixel)
        y_min_idx = int(y_min * pixel)
        y_max_idx = int(y_max * pixel)
    else:
        x_min_idx, x_max_idx, y_min_idx, y_max_idx = x_min, x_max, y_min, y_max

    return x_min_idx, x_max_idx, y_min_idx, y_max_idx


def create_mask_angle_tensor(pixel, bboxs, angles, scale=True):
    tensor = torch.zeros((2, pixel, pixel))

    for bbox, angle in zip(bboxs, angles):
        x_min_idx, x_max_idx, y_min_idx, y_max_idx = bbox_mask(pixel, bbox, scale)

        angle_without_direction = angle % 180
        angle_rad = math.radians(angle_without_direction*2)

        tensor[0, x_min_idx:x_max_idx, y_min_idx:y_max_idx] = math.sin(angle_rad)
        tensor[1, x_min_idx:x_max_idx, y_min_idx:y_max_idx] = math.cos(angle_rad)

    return tensor.permute(0, 2, 1)


def create_mask_tensor(pixel, bboxs, scale=True):
    tensor = torch.zeros((pixel, pixel))
    
    for bbox in bboxs:
        x_min_idx, x_max_idx, y_min_idx, y_max_idx = bbox_mask(pixel, bbox, scale)
        tensor[x_min_idx:x_max_idx, y_min_idx:y_max_idx] = 1

    return tensor.T


def create_mask_speed_tensor(pixel, bboxs, speeds):
    tensor = torch.zeros((1, pixel, pixel))

    for bbox, speed in zip(bboxs, speeds):
        x_min_idx, x_max_idx, y_min_idx, y_max_idx = bbox_mask(pixel, bbox, scale=True)
        tensor[0, x_min_idx:x_max_idx, y_min_idx:y_max_idx] = torch.tensor(speed, dtype=torch.float32)

    return tensor.permute(0, 2, 1)


# make the 2Dimage and target data 
class CNNData(Dataset):
    def __init__(self, inp_tar_list, pixel_per_axis):
        (self.other_vehicles_bboxs,
         self.other_vehicles_angles,
         self.objects_of_interest,
         self.targets,
         self.frame_timestamps,
         self.ids,
         self.angles) = zip(*inp_tar_list)

        self.pixel_per_axis = pixel_per_axis

    def __len__(self):
        return len(self.objects_of_interest)

    def __getitem__(self, idx):
        other_bboxs = self.other_vehicles_bboxs[idx]
        other_angles = self.other_vehicles_angles[idx]
        obj_bbox = self.objects_of_interest[idx]
        obj_angle = self.angles[idx]
        target = self.targets[idx]
        frame_ts = self.frame_timestamps[idx]
        obj_id = self.ids[idx]

        mask_tensor_others_angle = create_mask_angle_tensor(self.pixel_per_axis, other_bboxs, angles=other_angles)
        mask_tensor_interest = create_mask_angle_tensor(self.pixel_per_axis, [obj_bbox], angles=[obj_angle])

        model_input = torch.cat((mask_tensor_others_angle, mask_tensor_interest), dim=0).float()
        target = torch.tensor(target, dtype=torch.float32)

        return model_input, target, frame_ts, obj_id

    