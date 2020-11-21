import os
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from pathing import get_training_dir, get_testing_dir
import pdb

# Dataset that handels loading of gait_data. Params: dirpath- path of data directory
class GaitData(Dataset):
    def __init__(self, dirpath):
        overlap = 0.2  # Time overlap between samples
        num_samples = 4 * 40  # 4seconds at 40Hz
        overlap = int(num_samples * overlap)

        # Get all filenames for each file type (x, x_time, y, y_time)
        x_filenames = sorted(glob.glob(dirpath + "/*x.csv"))
        x_time_filenames = sorted(glob.glob(dirpath + "/*x_time.csv"))
        y_filenames = sorted(glob.glob(dirpath + "/*y.csv"))
        y_time_filenames = sorted(glob.glob(dirpath + "/*y_time.csv"))

        x_cols = ['x1','x2','x3','x4','x5','x6']

        data_df = []

        # Combine all files into one dataframe per type
        for x_fn, xt_fn, y_fn, yt_fn in zip(x_filenames, x_time_filenames, y_filenames, y_time_filenames):
            x_df = pd.read_csv(x_fn, dtype=np.float32, names=x_cols)
            xt_df = pd.read_csv(xt_fn, dtype=np.float32, names=['time'])
            y_df = pd.read_csv(y_fn, dtype=np.float32, names=['y'])
            yt_df = pd.read_csv(yt_fn, dtype=np.float32, names=['time'])

            x_combined_df = pd.concat([x_df, xt_df], axis=1)
            y_combined_df = pd.concat([y_df, yt_df], axis=1)

            # aligns time series data by filling left
            merged_x_y_df = pd.merge_asof(x_combined_df, y_combined_df, on='time')

            # assuming all NaN are in column y at the begining, fill them with y[0]
            data_df.append(merged_x_y_df.fillna(y_df.iat[0,0]))

        # Combine all files into one datatype for x and y
        x_data = []
        y_data = []

        # append all trials into one master dataframe
        for trial in data_df:
            x_data.append(np.array([trial[x_cols][i*(num_samples-overlap):i*(num_samples-overlap) + num_samples].to_numpy().flatten() for i in range(int(len(trial) / (num_samples-overlap))-1)]))
            y_data.append(np.array([trial.at[i*(num_samples-overlap)+int(num_samples/2),'y'] for i in range(int(len(trial) / (num_samples-overlap))-1)]))

        self.x = np.concatenate(x_data)
        self.y = np.concatenate(y_data)

    # get length of dataset
    def __len__(self):
        return (len(self.x))

    # Get specific instance of data
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self.x[idx], self.y[idx])


# get all of the data in the training folder without test/validation split
def get_training_dataloader(batch_size, kwargs):
    training_dir = get_training_dir()
    dataset = GaitData(dirpath=training_dir)
    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, **kwargs)
    return dataloader

# get all data from training folder and split into training and validation/testing sets
def get_train_test_dataset():
    training_dir = get_training_dir()
    dataset = GaitData(dirpath=training_dir)
    lengths = [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)]
    return random_split(dataset, lengths)


# take the splits from previous function and input them into dataloaders
def get_train_test_dataloaders(batch_size, kwargs):
    trainDS, testDS = get_train_test_dataset()

    trainloader = torch.utils.data.DataLoader(
            trainDS, batch_size=batch_size, shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(
            testDS, batch_size=batch_size, shuffle=True, **kwargs)
    return (trainloader, testloader)


# Get data in testing folder. Note not functional Issues with missing label data. Use training data as validation
def get_test_dataloader(batch_size, kwargs):
    test_dir = get_testing_dir()
    dataset = GaitData(dirpath=test_dir)
    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, **kwargs)
    return dataloader