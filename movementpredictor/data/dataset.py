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


log = logging.getLogger(__name__)


def get_path(num, path_store):
    if num % 3 == 0:
        path = path_store + "/train_cnn"
    elif num % 3 == 1:
        path = path_store + "/clustering"
    else:
        path = path_store + "/test"
    return path 


def getTorchDataSet(path_store, folder:str, num_dataset:int):
    path_frame = path_store + "/frame.pth"

    if folder == "train_cnn":
        path = path_store + "/train_cnn/dataset" + str(num_dataset*3) + ".pkl"
    elif folder == "clustering":
        path = path_store + "/clustering/dataset" + str(1+num_dataset*3) + ".pkl"
    elif folder == "test":
        path = path_store + "/test/dataset" + str(2+num_dataset*3) + ".pkl"
    else:
        log.error("no such folder " + folder)

    with open(path, "rb") as f:
        raw_dataset = pickle.load(f)
    #raw_dataset = torch.load(path)
    frame = torch.load(path_frame)

    torch_dataset = CNNData(frame, raw_dataset)

    return torch_dataset


def merge_and_split_datasets(path_data, val_split_ratio=0.01):
    merged_dataset = merge_datasets(path_data, "train_cnn")
    
    torch.manual_seed(42)
    total_size = len(merged_dataset)
    val_size = int(total_size * val_split_ratio)
    train_size = total_size - val_size
    
    train_ds, val_ds = random_split(merged_dataset, [train_size, val_size])
    
    return train_ds, val_ds


def merge_datasets(path_data, name="clustering"):
    datasets = []
    for part in range(4):
        ds = getTorchDataSet(path_data, name, part)
        datasets.append(ds)
    
    merged_dataset = ConcatDataset(datasets)
    return merged_dataset


def getTorchDataLoader(dataset, val_split=False, shuffle=True):
    batch_size = 8

    if val_split:
        torch.manual_seed(42)

        val_size = int(len(dataset)*0.02)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        return train_loader, val_loader
    
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def store_frame(frame: torch.Tensor, path_store):
    torch.save(frame, path_store + "/frame.pth")


def store_data(tracks: dict, path_store, num_batch):
    dataset = make_input_target_pairs(tracks)

    path = get_path(num_batch, path_store)
    os.makedirs(path, exist_ok=True)

    with open(path + "/dataset" + str(num_batch) + ".pkl", "wb") as f:
        pickle.dump(dataset, f)
    #torch.save(dataset, path + "/dataset" + str(num_batch) + ".pth")


def make_input_target_pairs(tracks: dict) -> Dataset:
    time_interval_in_millisec = 1100
    prediction_step = 1000
    input_target_pairs = []

    for trajectory in tqdm(tracks.values(), desc="creating dataset - calculating target positions"):
        if len(trajectory) == 0:
            continue

        for i, input_tr in enumerate(trajectory):
            timecur = input_tr.get_capture_ts()
            ts = []

            for next_tr in trajectory[i:]:
                ts.append(next_tr)
                if next_tr.get_capture_ts() - timecur > time_interval_in_millisec:
                    if len(ts) >= 4 and next_tr.get_capture_ts() - ts[-2].get_capture_ts() < 500:
                        input_target_pairs.append([input_tr, get_position_after_time(ts, prediction_step)])
                    break
    
    timestamp_dict = defaultdict(list)
    for _, trajectory in tracks.items():
        for track in trajectory:
            timestamp_dict[track.get_capture_ts()].append(track)

    for i, (inp, tar) in tqdm(enumerate(input_target_pairs), desc="creating dataset - collecting all bboxs"):
        other_vehicles = timestamp_dict[inp.get_capture_ts()]
        bboxs = [vehicle.get_bbox() for vehicle in other_vehicles]

        bboxs_tensor = np.array(bboxs, dtype=np.float32)
        input_tensor = np.array(inp.get_bbox(), dtype=np.float32)
        target_tensor = np.array(tar, dtype=np.float32)
        frame_ts = np.str_(inp.get_capture_ts())
        obj_id = np.str_(inp.get_uuid())
        angle = np.float32(inp.get_movement_angle())

        input_target_pairs[i] = [bboxs_tensor, input_tensor, target_tensor, frame_ts, obj_id, angle]
    
    return input_target_pairs


def get_position_after_time(ts, target_time):

    times = [track.get_capture_ts() for track in ts]
    xs = [track.get_center()[0] for track in ts]
    ys = [track.get_center()[1] for track in ts]
    
    target_time_stamp = ts[0].get_capture_ts() + target_time
    x_interp = np.interp(target_time_stamp, times, xs)
    y_interp = np.interp(target_time_stamp, times, ys)

    return [x_interp, y_interp]


def plotDataSamples(dataloader: DataLoader, amount: int):

    for count, (sample_batch, target_batch, _, _) in enumerate(dataloader):
        if count >= amount: break

        sample, target = sample_batch[0], target_batch[0]

        frame_np = sample[0].cpu().numpy()
        mask_others_np = sample[1].cpu().numpy()
        mask_interest_np_sin = sample[2].cpu().numpy()
        mask_interest_np_cos = sample[3].cpu().numpy()
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

        plt.savefig("plots/exampleInput" + str(count) + ".png")
        plt.close()


def create_mask_angle_tensors(dim_x, dim_y, bboxs, angle, scale=True):
    tensor = torch.zeros((2, dim_x, dim_y))
    
    for bbox in bboxs:
        [x_min, y_min], [x_max, y_max] = bbox

        if scale:
            x_min_idx = int(x_min * dim_x)
            x_max_idx = int(x_max * dim_x)
            y_min_idx = int(y_min * dim_y)
            y_max_idx = int(y_max * dim_y)
        else:
            x_min_idx, x_max_idx, y_min_idx, y_max_idx = x_min, x_max, y_min, y_max

        #if x_max_idx-x_min_idx < 1:
         #   x_max_idx = x_min_idx + 1
          #  x_min_idx = x_min_idx - 1
        #elif x_max_idx-x_min_idx < 2:
         #   x_max_idx = x_min_idx + 1
        
        #if y_max_idx-y_min_idx < 1:
         #   y_max_idx = y_min_idx + 1
          #  y_min_idx = y_min_idx - 1
        #elif y_max_idx-y_min_idx < 2:
         #   y_max_idx = y_min_idx + 1

        angle_without_direction = angle % 180
        angle_rad = math.radians(angle_without_direction*2)

        tensor[0, x_min_idx:x_max_idx, y_min_idx:y_max_idx] = math.sin(angle_rad)
        tensor[1, x_min_idx:x_max_idx, y_min_idx:y_max_idx] = math.cos(angle_rad)

    return tensor.permute(0, 2, 1)


def create_mask_tensor(dim_x, dim_y, bboxs, scale=True):
    tensor = torch.zeros((dim_x, dim_y))
    
    for bbox in bboxs:
        [x_min, y_min], [x_max, y_max] = bbox

        if scale:
            x_min_idx = int(x_min * dim_x)
            x_max_idx = int(x_max * dim_x)
            y_min_idx = int(y_min * dim_y)
            y_max_idx = int(y_max * dim_y)
        else:
            x_min_idx, x_max_idx, y_min_idx, y_max_idx = x_min, x_max, y_min, y_max

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
        self.other_vehicles = [item[0] for item in inp_tar_list]
        self.objects_of_interest = [item[1] for item in inp_tar_list]
        self.targets = [item[2] for item in inp_tar_list]
        self.frame_timestamps = [item[3] for item in inp_tar_list]
        self.ids = [item[4] for item in inp_tar_list]
        self.angles = [item[5] for item in inp_tar_list]

    def __len__(self):
        return len(self.objects_of_interest)

    def __getitem__(self, idx):

        inp_track = self.objects_of_interest[idx]
        other_bboxs = self.other_vehicles[idx]
        tar_pos = self.targets[idx]
        frame_ts = self.frame_timestamps[idx]
        obj_id = self.ids[idx]

        #frame_tensor = self.frames[inp_track.get_capture_ts()].unsqueeze(0)
        shape = (self.frame.shape[-1], self.frame.shape[-2])
        mask_tensor_others = create_mask_tensor(shape[0], shape[1], other_bboxs).unsqueeze(0)
        mask_tensors_interest = create_mask_angle_tensors(shape[0], shape[1], [inp_track], angle=self.angles[idx])
        sample = torch.cat((self.frame.unsqueeze(0), mask_tensor_others, mask_tensors_interest), dim=0).to(torch.float32)
        
        #target = create_target_tensor(shape[0], shape[1], tar_pos)
        target = torch.tensor(tar_pos).to(torch.float32)

        return sample, target, frame_ts, obj_id
