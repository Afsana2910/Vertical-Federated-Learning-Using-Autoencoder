import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier as gbm
import torch 

df_train = pd.read_csv('adult.csv')
df_train.drop_duplicates(keep='first',inplace=True)
df_train['income'].replace(['<=50K','>50K'],[0,1],inplace=True)
Y = df_train['income']
df_train = df_train.drop('income', axis=1)
df_client1 = df_train[["workclass","education","marital-status","age","fnlwgt"]]
df_client2 = df_train[["occupation","relationship","race","educational-num","capital-gain"]]
df_client3 = df_train[["gender","native-country","capital-loss","hours-per-week"]]

def client1_data():
    a = df_client1.iloc[:,[0,1,2,4]]
    b = df_client1.iloc[:,[3]]
    a_ohe = pd.get_dummies(a)
    a_ohe.shape, b.shape

    c = pd.concat([b, a_ohe], axis = 1)

    return c.values , Y.values

def client2_data():
    a = df_client2.iloc[:,[0,1,2]]
    b = df_client2.iloc[:,[3,4]]
    a_ohe = pd.get_dummies(a)
    a_ohe.shape, b.shape

    c = pd.concat([b, a_ohe], axis = 1)

    return c.values

def client3_data():
    a = df_client3.iloc[:,[0,1]]
    b = df_client3.iloc[:,[2,3]]
    a_ohe = pd.get_dummies(a)
    a_ohe.shape, b.shape

    c = pd.concat([b, a_ohe], axis = 1)

    return c.values


# X , Y = client1_data()
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# logr_pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='liblinear',max_iter=1000))
# logr_pipe.fit(X_train,y_train)
# y_pred = logr_pipe.predict(X_test)
# print(accuracy_score(y_test, y_pred))
# print(roc_auc_score(y_test, y_pred))