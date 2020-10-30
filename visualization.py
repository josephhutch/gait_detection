import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from pathing import *
from data_loaders import get_test_dataloader, get_training_dataloader
from VAE import VAE

def load_model(filename):
    path = os.path.join(get_model_dir(), filename)
    model = torch.load(path)
    model.to('cpu')
    return model.eval()


def in_out_comparison(filename, test=False):
    model = load_model(filename)
    if test:
        dataloader = get_test_dataloader(batch_size=1, kwargs={'num_workers': 1})
    else:
        dataloader = get_training_dataloader(batch_size=1, kwargs={'num_workers': 1})

    for data, labels in dataloader:
        r_batch, mu, logvar = model(data)
        t = np.linspace(0,4,4*40)
        x = data.reshape([6,160]).detach().numpy()
        x_r = r_batch.reshape([6,160]).detach().numpy()
        fig, ax = plt.subplots(nrows=2, ncols=1)
        for channel in x:
            ax[0].plot(t,channel)
        ax[0].set(xlabel='Time(s)', ylabel="Magnitude", title='Original Signal')
        for channel in x_r:
            ax[1].plot(t,channel)
        ax[1].set(xlabel='Time(s)', ylabel="Magnitude", title='Reconstructed Signal')
        plt.show()


def sample(filename):
    model = load_model(filename)
    dataloader = get_training_dataloader(batch_size=1, kwargs={'num_workers': 1})
    for data,l in dataloader:
        example_z = model.encode(data)
        break
    z = torch.rand([2,20])


if __name__ == '__main__':
    # in_out_comparison('300_epoch_basic.pt')
    sample('300_epoch_basic.pt')