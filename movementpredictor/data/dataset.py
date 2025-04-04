from typing import Dict
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
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


def getTorchDataSet(path_store, val_split_ratio=None):
    parent_folder = os.path.dirname(path_store.rstrip(os.sep))
    path_frame = os.path.join(parent_folder, "frame.pth")

    if os.path.exists(path_frame):
        frame = torch.load(path_frame)
    else:
        log.error(f"The file {path_frame} does not exist.")

    pkl_files = [f for f in os.listdir(path_store) if f.endswith(".pkl")]
    all_data = []

    for file in pkl_files:
        file_path = os.path.join(path_store, file)
        
        with open(file_path, "rb") as f:
            data = pickle.load(f)  
            all_data.append(data)  

    raw_dataset = [item for data_list in all_data for item in data_list]
    torch_dataset = CNNData(frame, raw_dataset)

    if val_split_ratio is not None:
        torch.manual_seed(42)
        total_size = len(torch_dataset)
        val_size = int(total_size * val_split_ratio)
        _, val_ds = random_split(torch_dataset, [total_size - val_size, val_size])
        return val_ds

    return torch_dataset


def getTorchDataLoader(dataset, shuffle=True):
    batch_size = 16
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def makeTorchDataLoader(tracks: Dict[str, list[TrackedObjectPosition]], path_frame: str, frame_rate) -> DataLoader:
    """
    Creates a pytorch DataLoader for make the data progressible for pytorch deep learning models.
    Based on a background_frame and the tracking information the model's input (frame containing boundingboxes) 
    and the target (vehicles position 1 second after input scene) are created and put into one dataset.
    
    Args:
        tracks: dict with trajectories (key: object id, value: list of this object's tracks)
        path_frame: path to the stored background frame that is the background information of all models inputs
        frame_rate: frame rate of video data
    
    Retruns: 
        DataLoader: pytorch Dataloader to make easy use of the dataset in batches

    """
    raw_dataset = make_input_target_pairs(tracks, frame_rate)
    frame = torch.load(path_frame)
    torch_dataset = CNNData(frame, raw_dataset)
    torch_dataloader = getTorchDataLoader(torch_dataset, shuffle=False)
    return torch_dataloader


def store_frame(frame: torch.Tensor, path_store1, path_store2):
    torch.save(frame, path_store1 + "/frame.pth")
    torch.save(frame, path_store2 + "/frame.pth")


def store_data(tracks: dict, path_store, frame_rate, num_batch=None):
    dataset = make_input_target_pairs(tracks, frame_rate)

    if num_batch is None:
        path = os.path.join(path_store, "test")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "dataset.pkl"), "wb") as f:
            pickle.dump(dataset, f)
        return

    if num_batch % 3 == 0:
        path = path_store + "/train_cnn"
    elif num_batch % 3 == 1:
        path = path_store + "/clustering"
    else:
        path = path_store + "/test"
    
    os.makedirs(path, exist_ok=True)

    with open(path + "/dataset" + str(num_batch) + ".pkl", "wb") as f:
        pickle.dump(dataset, f)


def make_input_target_pairs(tracks: dict, frame_rate: float) -> Dataset:
    time_interval_in_millisec = 1100
    prediction_step = 1000
    input_target_pairs = []

    for trajectory in tqdm(tracks.values(), desc="creating dataset - calculating target positions"):
        if len(trajectory) == 0:
            continue
        
        cars = [input_tr.class_id == 2 for input_tr in trajectory]
        if sum(cars) / len(cars) > 0.8:      # only examine vehicles that are at least 80% classified as a car -> high probability of actually being a car
            for i, input_tr in enumerate(trajectory):
                timecur = input_tr.capture_ts
                ts = []

                for next_tr in trajectory[i:]:
                    ts.append(next_tr)
                    if next_tr.capture_ts - timecur > time_interval_in_millisec:
                        # make sure the object is detected in at least 80% of the frames 
                        if len(ts) >= 0.8*(time_interval_in_millisec/1000)*frame_rate: #and next_tr.capture_ts - ts[-2].capture_ts < 500:
                            input_target_pairs.append([input_tr, get_position_after_time(ts, prediction_step)])
                        break
    
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


def plotDataSamples(dataloader: DataLoader, amount: int, path: str):

    for count, (sample_batch, target_batch, _, _) in enumerate(dataloader):
        if count >= amount: break

        sample, target = sample_batch[0], target_batch[0]

        frame_np = sample[0].cpu().numpy()

        mask_others_np_sin = sample[1].cpu().numpy()
        mask_others_np_cos = sample[2].cpu().numpy()
        mask_others_np = np.zeros(frame_np.shape)
        mask_others_np[(mask_others_np_sin != 0) | (mask_others_np_cos != 0)] = 1

        mask_interest_np_sin = sample[3].cpu().numpy()
        mask_interest_np_cos = sample[4].cpu().numpy()
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

        plt.savefig(os.path.join(path, "exampleInput" + str(count) + ".png"))
        plt.close()


def bbox_mask(dim_x, dim_y, bbox, scale):
    [x_min, y_min], [x_max, y_max] = bbox

    if scale:
        x_min_idx = int(x_min * dim_x)
        x_max_idx = int(x_max * dim_x)
        y_min_idx = int(y_min * dim_y)
        y_max_idx = int(y_max * dim_y)
    else:
        x_min_idx, x_max_idx, y_min_idx, y_max_idx = x_min, x_max, y_min, y_max
    
    #if x_max_idx - x_min_idx < 2:
     #   x_max_idx = x_max_idx + 1
      #  x_min_idx = x_max_idx - 2
    
    #if y_max_idx - y_min_idx < 2:
     #   y_max_idx = y_max_idx + 1
      #  y_min_idx = y_max_idx - 2

    return x_min_idx, x_max_idx, y_min_idx, y_max_idx


def create_mask_angle_tensor_(dim_x, dim_y, bboxs, angles, scale=True):
    tensor = torch.zeros((2, dim_x, dim_y))

    for bbox, angle in zip(bboxs, angles):
        x_min_idx, x_max_idx, y_min_idx, y_max_idx = bbox_mask(dim_x, dim_y, bbox, scale)

        angle_without_direction = angle % 180
        angle_rad = math.radians(angle_without_direction*2)

        tensor[0, x_min_idx:x_max_idx, y_min_idx:y_max_idx] = math.sin(angle_rad)
        tensor[1, x_min_idx:x_max_idx, y_min_idx:y_max_idx] = math.cos(angle_rad)

    return tensor.permute(0, 2, 1)


def create_mask_angle_tensor(dim_x, dim_y, bbox, angle, scale=True):
    tensor = torch.zeros((2, dim_x, dim_y))

    x_min_idx, x_max_idx, y_min_idx, y_max_idx = bbox_mask(dim_x, dim_y, bbox, scale)

    angle_without_direction = angle % 180
    angle_rad = math.radians(angle_without_direction*2)

    tensor[0, x_min_idx:x_max_idx, y_min_idx:y_max_idx] = math.sin(angle_rad)
    tensor[1, x_min_idx:x_max_idx, y_min_idx:y_max_idx] = math.cos(angle_rad)

    return tensor.permute(0, 2, 1)


def create_mask_tensor(dim_x, dim_y, bboxs, scale=True):
    tensor = torch.zeros((dim_x, dim_y))
    
    for bbox in bboxs:
        x_min_idx, x_max_idx, y_min_idx, y_max_idx = bbox_mask(dim_x, dim_y, bbox, scale)
        tensor[x_min_idx:x_max_idx, y_min_idx:y_max_idx] = 1

    return tensor.T


def create_target_tensor(dim_x, dim_y, tar_pos):
    x, y = tar_pos
    x, y = x*dim_x, y*dim_y
    target = torch.tensor([x, y]).to(torch.float32)
    return target

# make the 2Dimage and target data 
class CNNData(Dataset):

    def __init__(self, frame, inp_tar_list):
        self.frame = frame
        self.other_vehicles_bboxs = [item[0] for item in inp_tar_list]
        self.other_vehicles_angles = [item[1] for item in inp_tar_list]
        self.objects_of_interest = [item[2] for item in inp_tar_list]
        self.targets = [item[3] for item in inp_tar_list]
        self.frame_timestamps = [item[4] for item in inp_tar_list]
        self.ids = [item[5] for item in inp_tar_list]
        self.angles = [item[6] for item in inp_tar_list]

    def __len__(self):
        return len(self.objects_of_interest)

    def __getitem__(self, idx):

        inp_bbox = self.objects_of_interest[idx]
        other_bboxs = self.other_vehicles_bboxs[idx]
        other_angles = self.other_vehicles_angles[idx]
        tar_pos = self.targets[idx]
        frame_ts = self.frame_timestamps[idx]
        obj_id = self.ids[idx]
        angle = self.angles[idx]

        #frame_tensor = self.frames[inp_track.capture_ts].unsqueeze(0)
        shape = (self.frame.shape[-1], self.frame.shape[-2])
        #mask_tensor_others = create_mask_tensor(shape[0], shape[1], other_bboxs).unsqueeze(0)
        #mask_tensors_interest = create_mask_angle_tensor(shape[0], shape[1], inp_track, angle=self.angles[idx])
        mask_tensor_others = create_mask_angle_tensor_(shape[0], shape[1], other_bboxs, angles=other_angles)
        mask_tensors_interest = create_mask_angle_tensor_(shape[0], shape[1], [inp_bbox], angles=[angle])
        sample = torch.cat((self.frame.unsqueeze(0), mask_tensor_others, mask_tensors_interest), dim=0).to(torch.float32)
        
        #target = create_target_tensor(shape[0], shape[1], tar_pos)
        target = torch.tensor(tar_pos).to(torch.float32)

        return sample, target, frame_ts, obj_id
    