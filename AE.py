import torch
import torch.nn as nn
import torch.optim as optim
import math
torch.manual_seed(420)

class autoencoder(nn.Module):
    def __init__(self, input_size, rate):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, int(input_size*rate)),
            nn.ReLU(True),
            nn.Linear(int(input_size*rate), int(input_size*rate)),
            nn.ReLU(True),
            nn.Linear(int(input_size*rate), int(input_size*rate)),
            nn.ReLU(True)
            )
            # nn.Linear(64, 128),
            # nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(int(input_size*rate), int(input_size*rate)),
            nn.ReLU(True),
            nn.Linear(int(input_size*rate), int(input_size*rate)),
            nn.ReLU(True),
            nn.Linear(int(input_size*rate), input_size),
            nn.ReLU(True)
            )
            # nn.Linear(64, input_size),
            # nn.ReLU(True))
        self.double()

    def forward(self, x):
        x_hat = self.encoder(x)
        latent_data = x_hat.detach().numpy()
        x_hat = self.decoder(x_hat)
        return x_hat, latent_data



