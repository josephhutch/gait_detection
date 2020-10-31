import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from pathing import *
from data_loaders import get_test_dataloader, get_training_dataloader
from VAE import VAE

def load_model(filename, device):
    path = os.path.join(get_model_dir(), filename)
    model = torch.load(path, map_location=device)
    return model.eval()


def in_out_comparison(filename, test=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(filename, device)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(filename, device)

    dataloader = get_training_dataloader(batch_size=1, kwargs={'num_workers': 1})
    for data,l in dataloader:
        example_z = model.encode(data)
        break
    z = torch.rand([2,20])


def latentVisualization(filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(filename, device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    dataloader = get_training_dataloader(batch_size=1, kwargs=kwargs)

    z_x = []
    z_y = []
    z_z = []
    color = []
    colors = ['r','b','g','y']

    for data, label in dataloader:
        data = data.to(device)
        mu, logvar = model.encode(data)
        z = model.reparameterize(mu, logvar).tolist()[0]
        z_x.append(z[0])
        z_y.append(z[1])
        z_z.append(z[2])
        color.append(colors[int(label.item())])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z_x, z_y, z_z, c=color)
    plt.show()


if __name__ == '__main__':
    # in_out_comparison('300_epoch_basic.pt')
    # latentVisualization('latent_test.pt')
    sample('300_epoch_basic.pt')