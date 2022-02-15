import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier as gbm
import torch 
# def preprocessing():
#     df_train = pd.read_csv('adult.csv')
#     df_train.drop_duplicates(keep='first',inplace=True)

#     df_train.drop(labels=['capital-gain', 'capital-loss'], axis=1, inplace=True)
#     df_train['income'].replace(['<=50K','>50K'],[0,1],inplace=True)
#     df_train.drop(labels=['educational-num'], axis=1, inplace=True)

#     df_train['education'].replace(['HS-grad','Some-college','Prof-school'],['HighSchoolGrad','HighSchoolGrad','HighSchoolGrad'],inplace=True)
#     df_train['education'].replace(['Bachelors','Masters','Doctorate','Assoc-voc','Assoc-acdm'],['Graduated','Graduated','Graduated','Graduated','Graduated'],inplace=True)
#     df_train['education'].replace(['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th'],['NoDiploma','NoDiploma','NoDiploma','NoDiploma','NoDiploma','NoDiploma','NoDiploma','NoDiploma'],inplace=True)

#     df_train['workclass'].replace('?','Private',inplace=True)

#     df_train['workclass'].replace(['Self-emp-not-inc','Self-emp-inc'],['SelfEmployed','SelfEmployed'],inplace=True)
#     df_train['workclass'].replace(['Local-gov','State-gov','Federal-gov'],['GovtEmployed','GovtEmployed','GovtEmployed'],inplace=True)
#     df_train['workclass'].replace(['Without-pay','Never-worked'],['Unemployed','Unemployed'],inplace=True)

#     df_train['marital-status'].replace(['Married-civ-spouse','Married-spouse-absent', 'Married-AF-spouse'],['Married','Married','Married'],inplace=True)
#     df_train['marital-status'].replace(['Never-married','Divorced', 'Separated', 'Widowed'],['Single','Single','Single','Single'],inplace=True)

#     df_train = df_train[df_train['native-country'] == 'United-States']
#     df_train.drop(labels=['native-country'], axis=1, inplace=True)

#     df_train['occupation'].replace('?','Exec-managerial',inplace=True)
#     df_train['occupation'].replace(['Tech-support','Priv-house-serv', 'Protective-serv', 'Other-service'],['Service','Service','Service','Service'],inplace=True)
#     df_train['occupation'].replace(['Farming-fishing','Handlers-cleaners', 'Transport-moving', 'Machine-op-inspct','Adm-clerical','Craft-repair'],['Waged','Waged','Waged','Waged','Waged','Waged'],inplace=True)
#     df_train['occupation'].replace(['Exec-managerial','Prof-specialty', 'Sales', 'Armed-Forces'],['Salaried','Salaried','Salaried','Salaried'],inplace=True)


#     df_train['relationship'].replace(['Husband','Wife', 'Other-relative'],['Related','Related','Related'],inplace=True)
#     df_train['relationship'].replace(['Not-in-family','Own-child', 'Unmarried'],['Single','Single','Single'],inplace=True)

#     df_train['race'].replace(['Amer-Indian-Eskimo','Asian-Pac-Islander'],['Other','Other'],inplace=True)

#     df_train.drop(labels=['fnlwgt'], axis=1, inplace=True)


#     dummyVar=['workclass','education','marital-status','occupation','relationship','race','gender']
#     df_train = pd.get_dummies(df_train, columns=dummyVar, prefix_sep="-")

#     X = df_train.drop('income', axis=1)
#     y = df_train['income']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#     return X_train.values, X_test.values, y_train.values, y_test.values

def preprocessing():
    df_train = pd.read_csv('adult.csv')
    df_train.drop_duplicates(keep='first',inplace=True)
    df_train['income'].replace(['<=50K','>50K'],[0,1],inplace=True)
    y = df_train['income']

    a = df_train.iloc[:,[1,3,5,6,7,8,9,13]]
    b = df_train.iloc[:,[0,2,4,10,11,12]]

    a_ohe = pd.get_dummies(a)
    a_ohe.shape, b.shape

    c = pd.concat([b, a_ohe], axis = 1)


    return c.values, y.values


# X, Y = preprocessing()
# X_train, X_test, y_train, y_test =  train_test_split(X, Y, test_size=0.3, random_state=42)
# logr_pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='liblinear'))
# logr_pipe.fit(X_train,y_train)
# print(logr_pipe.score(X_test,y_test))
# y_pred = logr_pipe.predict(X_test)
# print(accuracy_score(y_test, y_pred))

# X, Y = preprocessing()
# print(X.shape)
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.metrics import roc_auc_score
# from mlxtend.feature_selection import SequentialFeatureSelector

# feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
#            k_features='50',
#            forward=False,
#            verbose=2,
#            scoring='roc_auc',
#            cv=4)

# features = feature_selector.fit(X, Y)
# print(features.k_feature_idx_)