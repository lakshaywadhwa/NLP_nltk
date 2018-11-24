# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:37:09 2018

@author: Lakshay Wadhwa
"""#######multiple_linear_regression

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
df=pd.read_csv("C:\\Users\\Lakshay Wadhwa\\Desktop\\wget\\FuelConsumptionCo2.csv", header = None, sep = ',', encoding = 'latin-1', error_bad_lines = False)
df.head()
df.describe()
cdf = df.loc[:,[4,5,10,12]]
X=df.loc[:,[4,5,10]]
y=df.loc[:,[12]]


from sklearn.linear_model import LinearRegression

X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')

X.fillna(0, inplace=True)
y.fillna(0, inplace=True)

from sklearn import datasets, linear_model, metrics 
 

  
# splitting X and y into training and testing sets 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
                                                    random_state=1) 
  
# create linear regression object 
reg = linear_model.LinearRegression() 
  
# train the model using the training sets 
reg.fit(X_train, y_train) 
  
# regression coefficients 
print('Coefficients: \n', reg.coef_) 
  
# variance score: 1 means perfect prediction 
print('Variance score: {}'.format(reg.score(X_test, y_test))) 
  
# plot for residual error 
  
## setting plot style 
plt.style.use('fivethirtyeight') 
  
## plotting residual errors in training data 
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, 
            color = "green", s = 10, label = 'Train data') 
  
## plotting residual errors in test data 
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, 
            color = "blue", s = 10, label = 'Test data') 
  
## plotting line for zero residual error 
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2) 
  
## plotting legend 
plt.legend(loc = 'upper right') 
  
## plot title 
plt.title("Residual errors") 
  
## function to show plot 
plt.show() 



from sklearn.metrics import r2_score


test_y_hat = reg.predict(X_test)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - y_test) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat ,y_test) )

