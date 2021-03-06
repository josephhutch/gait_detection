from torch import nn, optim
import torch
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # input 4 seconds of data at 40Hz over 6 channels (3 gyro, 3 accel)
        input_size = 4 * 40 * 6

        self.fc1 = nn.Linear(input_size, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        #return torch.sigmoid(self.fc4(h3))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class Hidden2VAE(nn.Module):
    def __init__(self):
        super(Hidden2VAE, self).__init__()

        # input 4 seconds of data at 40Hz over 6 channels (3 gyro, 3 accel)
        input_size = 4 * 40 * 6

        self.fc1 = nn.Linear(input_size, 600)
        self.fc2 = nn.Linear(600, 300)
        self.fc31 = nn.Linear(300, 20)
        self.fc32 = nn.Linear(300, 20)
        self.fc4 = nn.Linear(20, 300)
        self.fc5 = nn.Linear(300, 600)
        self.fc6 = nn.Linear(600, input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        #return torch.sigmoid(self.fc4(h3))
        return self.fc6(h5)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class Latent3VAE(nn.Module):
    def __init__(self):
        super(Latent3VAE, self).__init__()

        # input 4 seconds of data at 40Hz over 6 channels (3 gyro, 3 accel)
        input_size = 4 * 40 * 6

        self.fc1 = nn.Linear(input_size, 400)
        self.fc21 = nn.Linear(400, 3)
        self.fc22 = nn.Linear(400, 3)
        self.fc3 = nn.Linear(3, 400)
        self.fc4 = nn.Linear(400, input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        #return torch.sigmoid(self.fc4(h3))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
