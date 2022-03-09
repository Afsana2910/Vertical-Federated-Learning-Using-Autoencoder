import os
import torch
import numpy as np
import torch.nn as nn
from AE import autoencoder
def encoder_trainer(model,data,path,file_name,percent=0.1,num_epochs=1000,lr= 1e-2,batch_size=1000):
    # model.apply(init_weights)
    x_train = data[:int(data.shape[0]*percent),:]
    x_train,x_test=x_train[:int(x_train.shape[0]*0.75),:],x_train[int(x_train.shape[0]*0.75):,:]

    n_batch=int(x_train.shape[0]/batch_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    least_loss=np.inf

    for epoch in range(num_epochs):
        avg_loss=0
        for b in range(1,n_batch+1):
            model.train()
            output,code = model(torch.tensor(x_train[batch_size*(b-1):batch_size*b,:],dtype=torch.double))
            loss = criterion(output,torch.tensor(x_train[batch_size*(b-1):batch_size*b,:],dtype=torch.double))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss+=loss.item()
        
        model.eval()
        output,code = model(torch.tensor(x_test,dtype=torch.double))
        loss_val = criterion(output,torch.tensor(x_test,dtype=torch.double)).item()
        
        print(f'epoch [{epoch + 1}/{num_epochs}], Training loss:{avg_loss/n_batch: .4f}, Validation loss:{loss_val: .4f}')
        
        if epoch>0:
            if loss_val<least_loss :
                os.remove( path+file_name+'.pth')
                torch.save(model.state_dict(),  path+file_name+'.pth')
                least_loss=loss_val
                print(f'checkpoint {epoch} saved !')
        else:
            torch.save(model.state_dict(), path+file_name+'.pth')
    
    return int(data.shape[0]*percent)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)