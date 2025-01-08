import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2
import random
import pybase64
from itertools import islice
import logging

from visionapi.sae_pb2 import SaeMessage
from visionlib.pipeline.publisher import RedisPublisher
from visionlib import saedump


log = logging.getLogger(__name__)


def makeTorchTrainingDataSets(tracks: map, path_saedump):
    train_size = int(len(tracks)*0.5)
    batch_size = 8
    val_size = min(400, int(len(tracks)*0.1))

    random.seed(42)
    keys = list(tracks.keys())
    random.shuffle(keys)

    train_tracks = {k: tracks[k] for i, k in enumerate(keys) if i < train_size}
    val_tracks = {k: tracks[k] for i, k in enumerate(keys) if i > len(tracks)-val_size-1}
    test_tracks = {k: tracks[k] for i, k in enumerate(keys) if i > train_size and i < len(tracks)-val_size-1}

    train_dataset = extractDetections(train_tracks, path_saedump)
    val_dataset = extractDetections(val_tracks, path_saedump)
    test_dataset = extractDetections(test_tracks, path_saedump)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def makeTorchPredictionDataSet(tracks:dict, path_saedump):
    dataset = extractDetections(tracks, path_saedump)
    return DataLoader(dataset, batch_size=1)


def extractDetections(tracks: dict, path_saedump) -> Dataset:
    time_interval_in_millisec = 1500
    prediction_step = 1000
    input_target_pairs = []

    for trajectory in tqdm(tracks.values()):
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
    
    return CNNData(input_target_pairs, path_saedump)


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
        mask_np = sample[1].cpu().numpy()
        target = target.cpu().numpy()

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title("input")
        plt.imshow(frame_np, cmap='gray', interpolation='nearest')
        plt.imshow(mask_np, cmap='Reds', alpha=0.5, interpolation='nearest')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("target")
        frame_np = (frame_np * 255).astype(np.uint8)
        frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2RGB)
        cv2.circle(frame_rgb, [round(target[0]*frame_np.shape[1]), round(target[1]*frame_np.shape[0])], radius=4, color=(255, 0, 0), thickness=-1)
        plt.imshow(frame_rgb)
        plt.imshow(mask_np, cmap='Reds', alpha=0.5, interpolation='nearest')
        plt.axis('off')


        plt.savefig("plots/exampleInput" + str(count) + ".png")
        plt.close()


def create_mask_tensor(dim_x, dim_y, bbox):
    [x_min, y_min], [x_max, y_max] = bbox

    x_min_idx = int(x_min * dim_x)
    x_max_idx = int(x_max * dim_x)
    y_min_idx = int(y_min * dim_y)
    y_max_idx = int(y_max * dim_y)

    tensor = torch.zeros((dim_x, dim_y))
    tensor[x_min_idx:x_max_idx, y_min_idx:y_max_idx] = 1

    return tensor.T


def create_frame_tensor(dim_x, dim_y, track, path_saedump):
        sae_messages = saedump.message_splitter(open(path_saedump, 'r'))
        message = next(islice(sae_messages, track.get_frame_idx(), track.get_frame_idx()+1))
        event = saedump.Event.model_validate_json(message)
        proto_bytes = pybase64.standard_b64decode(event.data_b64)
        proto = SaeMessage()
        proto.ParseFromString(proto_bytes)

        frame = proto.frame
        if frame.timestamp_utc_ms != track.get_capture_ts():
            log.error("not the correct frame: \nframe timestamp: " + str(frame.timestamp_utc_ms) 
                      + "\ntrack timestamp: " + str(track.get_capture_ts()))

        frame_data = frame.frame_data_jpeg
        np_arr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)  

        resized_img = cv2.resize(img, (dim_x, dim_y), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized_img).float() / 255.0  
        return tensor


# make the 2Dimage and target data 
class CNNData(Dataset):

    dim_x = 280
    dim_y = 160

    def __init__(self, inp_tar_pairs, path_saedump):
        self.data = [pair[0] for pair in inp_tar_pairs]
        self.targets = [pair[1] for pair in inp_tar_pairs]
        self.path_saedump = path_saedump

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        inp_track = self.data[idx]
        mask_tensor = create_mask_tensor(self.dim_x, self.dim_y, inp_track.get_bbox()).unsqueeze(0)
        frame_tensor = create_frame_tensor(self.dim_x, self.dim_y, inp_track, self.path_saedump).unsqueeze(0)
        sample = torch.cat((frame_tensor, mask_tensor), dim=0).to(torch.float32)
        
        tar_pos = self.targets[idx]
        target = torch.tensor(tar_pos).to(torch.float32)

        return sample, target
