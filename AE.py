import torch
import torch.nn as nn
import torch.optim as optim
import math

class AE(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        #Encoder
        self.input_layer = nn.Linear(
            in_features=input_size, out_features=(int(input_size/2))
        )
        #torch.nn.init.uniform_(self.input_layer.weight)
        #torch.nn.init.uniform_(self.input_layer.bias)

        self.hidden_layer1 = nn.Linear(
            in_features=(int(input_size/2)), out_features=(int(input_size/2))
        )
        #torch.nn.init.uniform_(self.hidden_layer1.weight)
        #torch.nn.init.uniform_(self.hidden_layer1.bias)
        #Decoder
        # self.hidden_layer2 = nn.Linear(
        #     in_features=128, out_features=64
        # )
        #torch.nn.init.uniform_(self.hidden_layer2.weight)
        #torch.nn.init.uniform_(self.hidden_layer2.bias)

        self.output_layer = nn.Linear(
            in_features=(int(input_size/2)), out_features=input_size
        )
        #torch.nn.init.uniform_(self.output_layer.weight)
        #torch.nn.init.uniform_(self.output_layer.bias)
        self.double()
        #self.relu=nn.ReLU()

    def forward(self, features):
        
        x_hat=self.input_layer(features)
        x_hat=nn.functional.relu(x_hat)
        x_hat=self.hidden_layer1(x_hat)
        x_hat=nn.functional.relu(x_hat)
        latent_data = x_hat.detach().numpy()
        # x_hat=self.hidden_layer2(x_hat)
        # x_hat=nn.functional.relu(x_hat)
        x_hat=self.output_layer(x_hat)
        x_hat=nn.functional.relu(x_hat)
        return x_hat, latent_data



