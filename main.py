import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,roc_auc_score
from AE import autoencoder
import utils 
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVR
import adultincomedataset as ad
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import numpy as np
import verticalsplit as vs
import diabetesData as data
from sklearn.linear_model import SGDClassifier
import utils
import heartdiseasedataset_split as ht
import redwine_dataset as rd
import iotattack
#Get Training Data for Each Client
c1 , y = vs.client1_data()
c2 = vs.client2_data()
c3 = vs.client3_data()


#Initialize autoencoder class instances for training data of each client
 
model1 = autoencoder(input_size=c1.shape[1],rate=0.25)
model2 = autoencoder(input_size=c2.shape[1],rate=0.25)
model3 = autoencoder(input_size=c3.shape[1],rate=0.25)

path='C:/Users/P70077043/Documents/Autoencoder/heartDiseaseDataset/75percentcompression/shuffle2/'

train=True

if train:
    utils.encoder_trainer(model=model1,data=c1,path=path,file_name='c1',percent=0.25,num_epochs=500,lr= 1e-2,batch_size=50)
    utils.encoder_trainer(model=model2,data=c2,path=path,file_name='c2',percent=0.25,num_epochs=500,lr= 1e-2,batch_size=50)
    utils.encoder_trainer(model=model3,data=c3,path=path,file_name='c3',percent=0.25,num_epochs=500,lr= 1e-2,batch_size=50)
    

c1 , y = c1[int(c1.shape[0]*0.25):,:] , y[int(c1.shape[0]*0.25):] 
c2 = c2[int(c2.shape[0]*0.25):,:]
c3 = c3[int(c3.shape[0]*0.25):,:] 

#Train data for training model after aggregation 
c1_train, y_train = c1[:int(c1.shape[0]*0.75),:], y[:int(c1.shape[0]*0.75)] 
c2_train = c2[:int(c2.shape[0]*0.75),:]
c3_train = c3[:int(c3.shape[0]*0.75),:]


#Test data for testing model after aggregation 
c1_test, y_test = c1[int(c1.shape[0]*0.75):,:], y[int(c1.shape[0]*0.75):] 
c2_test = c2[int(c2.shape[0]*0.75):,:]
c3_test = c3[int(c3.shape[0]*0.75):,:]

model1.load_state_dict(torch.load(path+'c1.pth'))
_,en_c1=model1(torch.tensor(c1_train,dtype=torch.double))

model2.load_state_dict(torch.load(path+'c2.pth'))
_,en_c2=model2(torch.tensor(c2_train,dtype=torch.double))

model3.load_state_dict(torch.load(path+'c3.pth'))
_,en_c3=model3(torch.tensor(c3_train,dtype=torch.double))

model1.load_state_dict(torch.load(path+'c1.pth'))
_,en_c1_test=model1(torch.tensor(c1_test,dtype=torch.double))

model2.load_state_dict(torch.load(path+'c2.pth'))
_,en_c2_test=model2(torch.tensor(c2_test,dtype=torch.double))

model3.load_state_dict(torch.load(path+'c3.pth'))
_,en_c3_test=model3(torch.tensor(c3_test,dtype=torch.double))


merged_train = np.hstack((c1_train,en_c2,en_c3))
merged_test = np.hstack((c1_test,en_c2_test,en_c3_test))

merged_train_encoded = np.hstack((en_c1,en_c2,en_c3))
merged_test_encoded = np.hstack((en_c1_test,en_c2_test,en_c3_test))
#logr_pipe = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3)).fit(merged_train,y_train)

logr_pipe2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1200)).fit(merged_train_encoded,y_train)
y_pred2 = logr_pipe2.predict(merged_test_encoded)
print("Accuracy encoding client 1 data: ", accuracy_score(y_test, y_pred2))
print("ROC_AUC encoding client 1 data: ", roc_auc_score(y_test, y_pred2))

logr_pipe1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1200)).fit(merged_train,y_train)
y_pred1 = logr_pipe1.predict(merged_test)
print("Accuracy without encoding client 1 data: ", accuracy_score(y_test, y_pred1))
print("ROC_AUC without encoding client 1 data: ", roc_auc_score(y_test, y_pred1))

