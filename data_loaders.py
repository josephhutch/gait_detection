import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathing import get_training_dir, get_testing_dir


class GaitData(Dataset):
    def __init__(self, dirpath, limit=None):
        overlap = 0.2
        num_samples = 4 * 40  # 4seconds at 40Hz
        self.x = []
        self.x_times = []
        self.y = []
        self.y_times = []

        # will need to load data from all files
        files = os.listdir(dirpath)
        for i in range(0, len(files), 4):
            if limit is not None:
                if limit <= i/4:
                    break

            x, x_time, y, y_time = files[i:i+4]
            x_data = pd.read_csv(os.path.join(dirpath, x), dtype=np.float32, header=None)
            x_data = np.array(
            [x_data[i:i + num_samples].to_numpy().flatten() for i in range(int(len(x_data) / num_samples))])
            self.x.append(x_data)

            x_timestamps = pd.read_csv(os.path.join(dirpath, x_time), dtype=np.float32, header=None)
            x_timestamps = np.array(
                [x_timestamps[i:i + num_samples].to_numpy().flatten() for i in range(int(len(x_timestamps) / num_samples))])
            self.x_times.append(x_timestamps)

            y_data = pd.read_csv(os.path.join(dirpath, y), dtype=np.float32, header=None)
            y_data = np.array(
                [y_data[i:i + num_samples].to_numpy().flatten() for i in range(int(len(y_data) / num_samples))])
            self.y.append(y_data)

            y_timestamps = pd.read_csv(os.path.join(dirpath, y_time), dtype=np.float32, header=None)
            y_timestamps = np.array(
                [y_timestamps[i:i + num_samples].to_numpy().flatten() for i in range(int(len(y_timestamps) / num_samples))])
            self.y_times.append(y_timestamps)

    def __len__(self):
        return (len(self.x[1]))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[1][idx]


def get_training_dataloader(batch_size, kwargs):
    training_dir = get_training_dir()
    dataset = GaitData(dirpath=training_dir)
    dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, **kwargs)
    return dataloader

def get_test_dataloader(batch_size, kwargs):
    training_dir = get_testing_dir()
    dataset = GaitData(dirpath=training_dir)
    dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, **kwargs)
    return dataloader