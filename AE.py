import torch
import torch.nn as nn
import torch.optim as optim
import math

#OverComplete AutoEncoder
class AE(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        #Encoder
        self.input_layer = nn.Linear(
            in_features=input_size, out_features=64
        )
   

        self.hidden_layer1 = nn.Linear(
            in_features=64, out_features=128
        )
    
        #Decoder
        self.hidden_layer2 = nn.Linear(
            in_features=128, out_features=64
        )


        self.output_layer = nn.Linear(
            in_features=64, out_features=input_size
        )
      
        self.double()
  

    def forward(self, features):
        
        x_hat=self.input_layer(features)
        x_hat=nn.functional.relu(x_hat)
        x_hat=self.hidden_layer1(x_hat)
        x_hat=nn.functional.relu(x_hat)
        latent_data = x_hat.detach().numpy()
        x_hat=self.hidden_layer2(x_hat)
        x_hat=nn.functional.relu(x_hat)
        x_hat=self.output_layer(x_hat)
        x_hat=nn.functional.relu(x_hat)
        return x_hat, latent_data
    
    
#UnderComplete AutoEncoder
# class AE(nn.Module):
#     def __init__(self, input_size):
#         super().__init__()

#         ##Encoder
#         self.input_layer = nn.Linear(
#             in_features=input_size, out_features=(int(input_size/2))
#         )
   

#         self.hidden_layer1 = nn.Linear(
#             in_features=(int(input_size/2)), out_features=(int(input_size/2))
#         )
    
#         ##Decoder

#         self.output_layer = nn.Linear(
#             in_features=(int(input_size/2)), out_features=input_size
#         )
      
#         self.double()
  

#     def forward(self, features):
        
#         x_hat=self.input_layer(features)
#         x_hat=nn.functional.relu(x_hat)
#         x_hat=self.hidden_layer1(x_hat)
#         x_hat=nn.functional.relu(x_hat)
#         latent_data = x_hat.detach().numpy()
#         x_hat=self.output_layer(x_hat)
#         x_hat=nn.functional.relu(x_hat)
#         return x_hat, latent_data



