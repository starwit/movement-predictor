import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from tqdm import tqdm
import cv2


def makeTorchTrainingDataSets(tracks: map):
    train_size = int(len(tracks)*0.5)
    batch_size = 8
    val_size = min(400, int(len(tracks)*0.1))

    train_tracks = {k: v for i, (k, v) in enumerate(tracks.items()) if i < train_size}
    val_tracks = {k: v for i, (k, v) in enumerate(tracks.items()) if i > len(tracks)-val_size-1}
    test_tracks = {k: v for i, (k, v) in enumerate(tracks.items()) if i > train_size and i < len(tracks)-val_size-1}

    # add street background or point cloud of detections
    background = np.zeros((80, 80, 3), dtype=np.uint8)

    count = 0
    for trajectory in tqdm(train_tracks.values()):
        if count == 1000:
                break
        for track in trajectory:

            center = [round(track.get_center()[0] * 80), round(track.get_center()[1] * 80)]
            color = (127, 127, 127) 
            cv2.circle(background, center, radius=1, color=color, thickness=-1)

        count += 1

    train_dataset = extractDetections(train_tracks, background)
    val_dataset = extractDetections(val_tracks, background)
    test_dataset = extractDetections(test_tracks, background)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, background


def makeTorchPredictionDataSet(tracks:dict, background):
    dataset = extractDetections(tracks, background)
    return DataLoader(dataset, batch_size=1)


def extractDetections(tracks: dict, background) -> Dataset:

    time_interval_in_sec = 1.2
    interpolated_trajectories = []
    timestamp_interval = 0.2
    input_target_pairs = []

    for trajectory in tqdm(tracks.values()):
        if len(trajectory) == 0:
            continue
        
        ts = [trajectory[0]]
        timeprev = trajectory[0].get_capture_ts().timestamp()
        for t in trajectory[1:]:

            timecur = t.get_capture_ts().timestamp()
            if timecur - timeprev > time_interval_in_sec:
                if len(ts) >= 2:
                    interpolated_trajectories.append(interpolate(ts, timestamp_interval))
                ts = []
                continue

            ts.append(t)
            timeprev = timecur
        
        if len(ts) >= 2:
            interpolated_trajectories.append(interpolate(ts, timestamp_interval))
        ts = []

    step = round(0.6/timestamp_interval)
    for trajectory in interpolated_trajectories:
        for ind in range(len(trajectory)-step):
            input_target_pairs.append([trajectory[ind], trajectory[ind+step]])
    
    return ConvAEData(input_target_pairs, background)


def interpolate(ts, time_interval):

    times = [track.get_capture_ts().timestamp() for track in ts]
    xs = [track.get_center()[0] for track in ts]
    ys = [track.get_center()[1] for track in ts]

    times_interp = np.linspace(times[0], times[-1], round((times[-1]-times[0]) / time_interval))
    x_interp = np.interp(times_interp, times, xs)
    y_interp = np.interp(times_interp, times, ys)

    interpTrajectory = [[x, y] for x, y in zip(x_interp, y_interp)]
    return interpTrajectory


def plotDataSamples(dataloader: DataLoader, amount: int, background):
    for count, (sample, target) in enumerate(dataloader):
        if count >= amount: break
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(sample[0].permute(1, 2, 0), np.uint8))
        plt.title("input")
        plt.subplot(1, 2, 2)
        img = background.copy()
        cv2.circle(img, [round(target[0][0].numpy()*80), round(target[0][1].numpy()*80)], radius=1, color=(255, 0, 0), thickness=-1)
        plt.imshow(np.array(img, np.uint8))
        #plt.imshow(img)
        plt.title("target")
        plt.savefig("aesanomalydetection/cnn/plots/inputAE" + str(count) + ".png")
        plt.close()


# make the trajectories 2Dimage data 
class ConvAEData(Dataset):

    def __init__(self, inp_tar_pairs, background):
        data = []
        targets = []

        for pair in inp_tar_pairs:
            targets.append(pair[1])
            img = background.copy()
            cv2.circle(img, [round(pair[0][0]*80), round(pair[0][1]*80)], radius=1, color=(255, 0, 0), thickness=-1)
            data.append(np.array(img, np.uint8))

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(np.array(self.data[idx])).permute(2, 0, 1).to(torch.float32)
        target = torch.tensor(self.targets[idx]).to(torch.float32)
        return sample, target
