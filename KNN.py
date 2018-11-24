# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 20:20:10 2018

@author: Lakshay Wadhwa
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

df=pd.read_csv("C:\\Users\\Lakshay Wadhwa\\Desktop\\wget\\teleCust1000t.csv", header = None, sep = ',', encoding = 'latin-1', error_bad_lines = False)
#########code for cleaning data for histogram and other plots
df[11].value_counts()
df[5] = df[5].apply(pd.to_numeric, errors='coerce')
df[5].fillna(0, inplace=True)
df[5].hist(bins=50)

X=df.loc[:,[0,1,2,3,4,5,6,7,8,9,10]]
y=df.loc[:,[11]]


X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')

X.fillna(0, inplace=True)
y.fillna(0, inplace=True)



X=X.drop(X.index[[0]])
y=y.drop(y.index[[0]])
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

from sklearn.neighbors import KNeighborsClassifier

k = 10
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

yhat = neigh.predict(X_test)
#accuracy evaluation
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))



###########iteration to see best k


Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
    
