import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pdb
from data_loaders import get_training_dataloader, get_test_dataloader, get_train_test_dataloaders
from pathing import *
from VAE import VAE, Latent3VAE

# Parse Args

parser = argparse.ArgumentParser(description='VAE Gait Data')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")


# Create Data Loader

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

(train_loader, test_loader) = get_train_test_dataloaders(batch_size=args.batch_size, kwargs=kwargs)
# train_loader = get_training_dataloader(batch_size=args.batch_size, kwargs=kwargs)

# Create Model

# model = VAE().to(device)
model = Latent3VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def loss_function(recon_x, x, mu, logvar):
    # MSE = 0.005 * F.mse_loss(recon_x, x, reduction='sum')
    # Play with turning the weight of MSE, 0.005 might be too small
    MSE = F.mse_loss(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD

def calc_error(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='sum').item()

def train(epoch):
    model.train()
    train_loss = 0
    error = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            error += calc_error(recon_batch, data)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    return np.sqrt(error / len(train_loader.dataset) / 960)


def test(epoch):
    model.eval()
    test_loss = 0
    error = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            error += calc_error(recon_batch, data)
            # if i == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                           recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            #     save_image(comparison.cpu(),
            #              'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    return np.sqrt(error / len(test_loader.dataset) / 960)

if __name__ == "__main__":
    train_error = []
    test_error = []
    for epoch in range(1, args.epochs + 1):
        train_error.append(train(epoch))
        test_error.append(test(epoch))
        # with torch.no_grad():
        #     sample = torch.randn(64, 20).to(device)
        #     sample = model.decode(sample).cpu()
        #     save_image(sample.view(64, 1, 28, 28),
        #                'results/sample_' + str(epoch) + '.png')

    plt.plot(train_error, label='Train Error')
    plt.plot(test_error, label='Validation Error')
    plt.title('Train and Validation Error while Training')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()
    torch.save(model, os.path.join(get_model_dir(), 'latent_test.pt'))