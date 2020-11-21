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
from VAE import VAE, Latent3VAE, Hidden2VAE

# Parse Args
# batch_size: number of samples the model views at a time
# epochs: number of training runs through all data
# no-cuda: dont use gpu
# seed: random seed
# log interval: how many batches before printing results in each epoch
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
model = Hidden2VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# loss function for training VAE
def loss_function(recon_x, x, mu, logvar):
    # MSE = 0.005 * F.mse_loss(recon_x, x, reduction='sum')
    # Reconstruction loss
    MSE = F.mse_loss(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # KL Divergence Loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD

# Error Metric for evaluating a training session
def calc_error(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='sum').item()

# training session
def train(epoch):
    # Initialize model to train and begin to record error and losses
    model.train()
    train_loss = 0
    error = 0
    # loop through training data
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()  # reset gradients
        recon_batch, mu, logvar = model(data)  # feed data through model
        loss = loss_function(recon_batch, data, mu, logvar)  # calculate loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()  # Backpropagate
        if batch_idx % args.log_interval == 0: # Print out losses for training intervals
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    with torch.no_grad():  # calculate reconstruction error
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            error += calc_error(recon_batch, data)

    # print avg loss for the epoch
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    return np.sqrt(error / len(train_loader.dataset) / 960)

# evaluate model on test/validation set
def test(epoch):
    # Initialize evaluation and error metric
    model.eval()
    test_loss = 0
    error = 0
    #loop through testing data
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)  # feed val data into model
            test_loss += loss_function(recon_batch, data, mu, logvar).item()  # calculate val loss
            error += calc_error(recon_batch, data)  # calculate validation recontruction error
            # if i == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                           recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            #     save_image(comparison.cpu(),
            #              'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    # Print out avg loss for this epoch
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    return np.sqrt(error / len(test_loader.dataset) / 960)

# Main training loop
if __name__ == "__main__":
    # error lists for plotting after training
    train_error = []
    test_error = []
    # training epochs
    for epoch in range(1, args.epochs + 1):
        train_error.append(train(epoch))  # train
        test_error.append(test(epoch))  # test

        # See sample of latent space. Only used for testing
        # with torch.no_grad():
        #     sample = torch.randn(64, 20).to(device)
        #     sample = model.decode(sample).cpu()
        #     save_image(sample.view(64, 1, 28, 28),
        #                'results/sample_' + str(epoch) + '.png')

    # plot results of training
    plt.plot(train_error, label='Train Error')
    plt.plot(test_error, label='Validation Error')
    plt.title('Train and Validation Error while Training')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()
    # save model
    torch.save(model, os.path.join(get_model_dir(), 'big_test.pt'))