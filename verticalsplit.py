import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier as gbm
import torch 

'''Using the Adult Income Dataset from the UCI Repository'''


df_train = pd.read_csv('C:/Users/P70077043/Documents/Autoencoder/adultIncomeDataset/shuffle1.csv')
# df_train = df_train[np.random.default_rng(seed=42).permutation(df_train.columns.values)]
# df_train.to_csv('C:/Users/P70077043/Documents/Autoencoder/adultIncomeDataset/shuffle3.csv',index=False)
Y = df_train['income']
df_train = df_train.drop('income', axis=1)
features = df_train.columns.tolist()
f1 = features[:5]
f2 = features[5:10]
f3 = features[10:]
df_client1 = df_train[f1]
df_client2 = df_train[f2]
df_client3 = df_train[f3]

# df_train['native-country'] = df_train['native-country'].replace('?',np.nan) 
# df_train['workclass'] = df_train['workclass'].replace('?',np.nan) 
# df_train['occupation'] = df_train['occupation'].replace('?',np.nan)
# df_train.dropna(how='any',inplace=True)
# df_train.drop_duplicates(keep='first',inplace=True)
# df_train['income'].replace(['<=50K','>50K'],[0,1],inplace=True)
# sample1 = df_train[df_train['income'] == 0]
# sample2 = df_train[df_train['income'] == 1]
# sample1 = sample1.head(sample2.shape[0])
# df_train = pd.concat([sample1, sample2], axis=0)
# df_train = df_train.sample(frac=1)
# df_train.to_csv("ProcessedAdultIncome.csv",index=False)
# Y = df_train['income']
# df_train = df_train.drop('income', axis=1)
# df_client1 = df_train[["workclass","education","marital-status","age","fnlwgt"]]
# df_client2 = df_train[["occupation","relationship","race","educational-num","capital-gain"]]
# df_client3 = df_train[["gender","native-country","capital-loss","hours-per-week"]]

def client1_data(flag=0):
    # a = df_client1.iloc[:,[0,1,2,4]]
    # b = df_client1.iloc[:,[3]]
    a = df_client1.select_dtypes(exclude=[np.number])
    b = df_client1.select_dtypes(include=[np.number])
    a_ohe = pd.get_dummies(a)
    a_ohe.shape, b.shape

    c = pd.concat([b, a_ohe], axis = 1)
    if flag==0:
        return c.values , Y.values
    else:
        return c.values

def client2_data():
    # a = df_client2.iloc[:,[0,1,2]]
    # b = df_client2.iloc[:,[3,4]]
    a = df_client2.select_dtypes(exclude=[np.number])
    b = df_client2.select_dtypes(include=[np.number])
    a_ohe = pd.get_dummies(a)
    a_ohe.shape, b.shape

    c = pd.concat([b, a_ohe], axis = 1)

    return c.values

def client3_data():
    # a = df_client3.iloc[:,[0,1]]
    # b = df_client3.iloc[:,[2,3]]
    a = df_client3.select_dtypes(exclude=[np.number])
    b = df_client3.select_dtypes(include=[np.number])
    a_ohe = pd.get_dummies(a)
    a_ohe.shape, b.shape

    c = pd.concat([b, a_ohe], axis = 1)

    return c.values
#logr_pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty='elasticnet', solver='saga',max_iter=1500,l1_ratio=0.5))

# merged_train = np.hstack((client1_data(1),client2_data(),client3_data()))
# X_train, X_test, y_train, y_test = train_test_split(merged_train, Y, test_size=0.25, random_state=42)
# logr_pipe = make_pipeline(StandardScaler(), LogisticRegression())
# logr_pipe.fit(X_train,y_train)
# y_pred = logr_pipe.predict(X_test)
# print("Accuracy in Centralized Training: ",accuracy_score(y_test, y_pred))
# print("ROC_AUC in Centralized Training: ",roc_auc_score(y_test, y_pred))

# X , y =client1_data()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# logr_pipe = make_pipeline(StandardScaler(), LogisticRegression())
# logr_pipe.fit(X_train,y_train)
# y_pred = logr_pipe.predict(X_test)
# print("Accuracy of Client 1: ",accuracy_score(y_test, y_pred))
# print("ROC_AUC of Client 1: ",roc_auc_score(y_test, y_pred))
