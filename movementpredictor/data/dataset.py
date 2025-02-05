import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2
import logging
import os
import gc
from collections import defaultdict


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
        path = path_store + "/train_cnn/dataset" + str(num_dataset*3) + ".pth"
    elif folder == "clustering":
        path = path_store + "/clustering/dataset" + str(1+num_dataset*3) + ".pth"
    elif folder == "test":
        path = path_store + "/test/dataset" + str(2+num_dataset*3) + ".pth"
    else:
        log.error("no such folder " + folder)

    raw_dataset = torch.load(path)
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


def getTorchDataLoader(dataset, val_split=False, train=True):
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
        return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=True)


def store_frame(frame: torch.Tensor, path_store):
    torch.save(frame, path_store + "/frame.pth")


def store_data(tracks: dict, path_store, num_batch):
    dataset = make_input_target_pairs(tracks)

    path = get_path(num_batch, path_store)
    os.makedirs(path, exist_ok=True)

    torch.save(dataset, path + "/dataset" + str(num_batch) + ".pth")


def make_input_target_pairs(tracks: dict) -> Dataset:
    time_interval_in_millisec = 1500
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
                    if len(ts) >= 2:
                        input_target_pairs.append([input_tr, get_position_after_time(ts, prediction_step)])
                    break
    
    timestamp_dict = defaultdict(list)
    for _, trajectory in tracks.items():
        for track in trajectory:
            timestamp_dict[track.get_capture_ts()].append(track)

    for i, (inp, tar) in tqdm(enumerate(input_target_pairs), desc="creating dataset - collecting all bboxs"):
        other_vehicles = timestamp_dict[inp.get_capture_ts()]
        bboxs = [vehicle.get_bbox() for vehicle in other_vehicles]

        bboxs_tensor = torch.Tensor(bboxs).to(torch.float32)
        input_tensor = torch.Tensor(inp.get_bbox()).to(torch.float32)
        target_tensor = torch.Tensor(tar).to(torch.float32)

        input_target_pairs[i] = [bboxs_tensor, input_tensor, target_tensor]
    
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

    for count, (sample_batch, target_batch) in enumerate(dataloader):
        if count >= amount: break

        sample, target = sample_batch[0], target_batch[0]

        frame_np = sample[0].cpu().numpy()
        mask_others_np = sample[1].cpu().numpy()
        mask_interest_np = sample[2].cpu().numpy()

        target = target.cpu().numpy()

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title("input")
        plt.imshow(frame_np, cmap='gray', interpolation='nearest')
        plt.imshow(mask_others_np, cmap='Reds', alpha=0.4, interpolation='nearest')
        plt.imshow(mask_interest_np, cmap='Blues', alpha=0.3, interpolation='nearest')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("target")
        frame_np = (frame_np * 255).astype(np.uint8)
        frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2RGB)
        cv2.circle(frame_rgb, [round(target[0]), round(target[1])], radius=4, color=(255, 0, 0), thickness=-1)
        plt.imshow(frame_rgb)
        plt.imshow(mask_others_np, cmap='Reds', alpha=0.4, interpolation='nearest')
        plt.imshow(mask_interest_np, cmap='Blues', alpha=0.3, interpolation='nearest')
        plt.axis('off')

        plt.savefig("plots/exampleInput" + str(count) + ".png")
        plt.close()


def create_mask_tensor(dim_x, dim_y, bboxs):
    tensor = torch.zeros((dim_x, dim_y))

    for bbox in bboxs:
        [x_min, y_min], [x_max, y_max] = bbox

        x_min_idx = int(x_min * dim_x)
        x_max_idx = int(x_max * dim_x)
        y_min_idx = int(y_min * dim_y)
        y_max_idx = int(y_max * dim_y)

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

    def __len__(self):
        return len(self.objects_of_interest)

    def __getitem__(self, idx):

        inp_track = self.objects_of_interest[idx]
        other_bboxs = self.other_vehicles[idx]
        tar_pos = self.targets[idx]

        #frame_tensor = self.frames[inp_track.get_capture_ts()].unsqueeze(0)
        shape = (self.frame.shape[-1], self.frame.shape[-2])
        mask_tensor_others = create_mask_tensor(shape[0], shape[1], other_bboxs).unsqueeze(0)
        mask_tensor_interest = create_mask_tensor(shape[0], shape[1], [inp_track]).unsqueeze(0)
        sample = torch.cat((self.frame.unsqueeze(0), mask_tensor_others, mask_tensor_interest), dim=0).to(torch.float32)
        
        
        target = create_target_tensor(shape[0], shape[1], tar_pos)

        return sample, target
