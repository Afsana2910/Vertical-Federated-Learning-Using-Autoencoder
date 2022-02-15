import torch
import torch.nn as nn
import torch.optim as optim
import math

class Train:
    def __init__(self, data, model):
        self.x =  data
        self.model = model
        self.lr = 1e-3
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),self.lr)
        self.output = None
        self.latentdata = None
        self.loss = None

    def get_latentdata(self):
        num_epochs = 100
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            self.output, self.latentdata = self.model(self.x)                        
    
            self.loss  = self.criterion(self.output, self.x)
            #************************ backward *************************
            self.loss.backward()
            self.optimizer.step()
            # ***************************** log ***************************
            #print(f'epoch [{epoch + 1}/{num_epochs}], loss:{self.loss.item(): .4f}')
        return self.latentdata



