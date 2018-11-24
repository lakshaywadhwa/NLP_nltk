# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 22:34:34 2018

@author: Lakshay Wadhwa
"""
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
df=pd.read_csv("C:\\Users\\Lakshay Wadhwa\\Desktop\\wget\\drug200.csv", header = None, sep = ',', encoding = 'latin-1', error_bad_lines = False)
X=df.loc[:,[0,1,2,3,4]]
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
#some features in this dataset are categorical such as Sex or BP. Unfortunately, Sklearn Decision Trees do not handle categorical variables. But still we can convert these features to numerical values. pandas.get_dummies() Convert categorical variable into dummy/indicator variables.
X=X.drop(X.index[[0]])
X.iloc[:,1] = le_sex.transform(X.iloc[:,1]) 

#X = X.apply(pd.to_numeric, errors='coerce')


#X.fillna(0, inplace=True)
#

le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X.iloc[:,2] = le_BP.transform(X.iloc[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X.iloc[:,3] = le_Chol.transform(X.iloc[:,3]) 

##We have successfully changed the ctaegorical values to numerical ones
y = df.loc[:,[5]]
y=y.drop(y.index[[0]])

from sklearn.model_selection import train_test_split

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters

#Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.
#Next, we will fit the data with the training feature matrix X_trainset and training response vector y_trainset

drugTree.fit(X_trainset,y_trainset)
predTree = drugTree.predict(X_testset)

print (predTree [0:5])
print (y_testset [0:5])

#Next, let's import metrics from sklearn and check the accuracy of our model.
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
#Accuracy classification score computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.

#In multilabel classification, the function returns the subset accuracy. If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.

#visualization
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

dot_data = StringIO()
filename = "drugtree.png"
#featureNames = df.loc[:,[0,1,2,3,4]]
#targetNames = df.loc[:,[5]].unique().tolist()
featureNames =df.columns[0:5]
targetNames = df.loc[:,[5]].values.tolist()

out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 

graph.write_png(filename)

img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')
