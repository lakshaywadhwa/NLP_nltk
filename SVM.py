# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 23:42:15 2018

@author: Lakshay Wadhwa
"""

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt
df=pd.read_csv("C:\\Users\\Lakshay Wadhwa\\Desktop\\wget\\cell_samples.csv", header = None, sep = ',', encoding = 'latin-1', error_bad_lines = False)
df.dtypes

#cleaning data
df = df.loc[pd.to_numeric(df.loc[:,6], errors='coerce').notnull()]
df[6] = df[6].astype('int')
df.dtypes
X = df.loc[:,[1,2,3,4,5,6,7,8,9]]

X = X.apply(pd.to_numeric, errors='coerce')


X.fillna(0, inplace=True)



y=df.loc[:,[10]]
y = y.apply(pd.to_numeric, errors='coerce')
y.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
#The SVM algorithm offers a choice of kernel functions for performing its processing. Basically, mapping data into a higher dimensional space is called kernelling. The mathematical function used for the transformation is known as the kernel function, and can be of different types, such as:

#1.Linearra
#2.Polynomial
#3.Radial basis function (RBF)
#4.Sigmoid

from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train,  np.ravel(y_train,order='C')) 
yhat = clf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print (classification_report(y_test, yhat))
print (confusion_matrix(y_test, yhat))


from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted') 


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)