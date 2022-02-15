import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error,accuracy_score,roc_auc_score
import scipy as sp 
import redwine_dataset as rd
sp.random.seed(42) 
import AE as autoencoder
import train_autoencoder 
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import adultincomedataset as ad
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import numpy as np
import verticalsplit as vs


c1 , y = vs.client1_data()
c2 = vs.client2_data()
c3 = vs.client3_data()
model1 = autoencoder.AE(input_size=c1.shape[1])
model2 = autoencoder.AE(input_size=c2.shape[1])
model3 = autoencoder.AE(input_size=c3.shape[1])

encode1 = train_autoencoder.Train(torch.tensor(c1,dtype=torch.double), model1)
encode2 = train_autoencoder.Train(torch.tensor(c2,dtype=torch.double), model2)
encode3 = train_autoencoder.Train(torch.tensor(c3,dtype=torch.double), model3)


c1_latent = encode1.get_latentdata()
c2_latent = encode2.get_latentdata()
c3_latent = encode3.get_latentdata()


merged = np.hstack((c1_latent,c2_latent,c3_latent))

x_train, x_test, y_train, y_test = train_test_split(merged, y, test_size=0.3, random_state=42)


acc = []
rocauc = []
for i in range(10):


    logr_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000,penalty='l1', solver='liblinear'))
    logr_pipe.fit(x_train,y_train)
    y_pred = logr_pipe.predict(x_test)
    acc.append(accuracy_score(y_test, y_pred))
    rocauc.append(roc_auc_score(y_test, y_pred))
 
    
print("Accuracy: ", max(acc))
print("ROCAUC: ", max(rocauc))


