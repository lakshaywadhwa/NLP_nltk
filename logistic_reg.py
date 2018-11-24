# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 21:10:01 2018

@author: Lakshay Wadhwa
"""

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt

df=pd.read_csv("C:\\Users\\Lakshay Wadhwa\\Desktop\\wget\\ChurnData.csv", header = None, sep = ',', encoding = 'latin-1', error_bad_lines = False)
df.head()



X=df.loc[:,[0,1,2,3,4,5,6]]

y=df.loc[:,[27]]

X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')

X.fillna(0, inplace=True)
y.fillna(0, inplace=True)


#########
 df.as_matrix(columns=[df[1:]])
 #############
 
 
 
X=X.drop(X.index[[0]])
y=y.drop(y.index[[0]])
X = preprocessing.StandardScaler().fit(X).transform(X)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

## define jaccard as the size of the intersection divided by the size of the union of two label sets. If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.

from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)
from sklearn.metrics import classification_report, confusion_matrix
confusion_matrix(y_test, yhat)
confusion_matrix(y_test, yhat, labels=[1,0])
print (classification_report(y_test, yhat))
