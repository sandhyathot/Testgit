# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 13:58:27 2018

@author: SANDHYA RANI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reaing the dataset and dropping the not required columns
dataset=pd.read_csv('carPrice.csv')
d1=dataset.drop(['car_ID','symboling','enginelocation','wheelbase','carlength','carwidth',
             'carheight','curbweight','boreratio','stroke','compressionratio','peakrpm',
             'citympg','highwaympg'], axis=1)
# code written in line 12 can be witten in betterway

#Categerize the pricecolumn and create a new column(Y)
def dependent(n):
    for i in range (1,n+1):
        if i<=10000:
            j = 0
        else:
            j = 1
    return j
y_depen=list(map(dependent, dataset['price'].values))
d1['y_depend']=y_depen

# X and Y values
X=d1.iloc[: ,:-1].values
Y=d1.iloc[:,-1].values

#Convert the strings into numerical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,0]  = labelencoder.fit_transform(X[:,0])
X[:,1]  = labelencoder.fit_transform(X[:,1])
X[:,2]  = labelencoder.fit_transform(X[:,2])
'''print (X[:,0],X[:,1],X[:,2])'''
onehotencoder = OneHotEncoder(categorical_features = [0,1,2])
X = onehotencoder.fit_transform(X).toarray()

'''x=pd.get_dummies(d1,columns=['doornumber','carbody','drivewheel'],drop_first=True).head()'''

# Split the data into test and train sets
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.15,random_state=20)

#apply standard scaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#apply logistic regression to the model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=20)
classifier.fit(X_train,Y_train)

#Predict the Y value 
Y_pred = classifier.predict(X_test)

#apply confusion matrix to themodel
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
print (cm)

