# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 19:24:17 2018

@author: Lakshay Wadhwa
"""

import random 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs
np.random.seed(0)
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
plt.scatter(X[:, 0], X[:, 1], marker='.')
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
k_means.fit(X)
k_means_labels = k_means.labels_
k_means_labels

k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers


#########applying the algorithm over the dataset
df=pd.read_csv("C:\\Users\\Lakshay Wadhwa\\Desktop\\wget\\Cust_segmentation.csv", header = None, sep = ',', encoding = 'latin-1', error_bad_lines = False)
df.dtypes
df = df.drop(df.loc[:,[8]], axis=1)
df.head()


from sklearn.preprocessing import StandardScaler
X = df.loc[:,1:]
X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet


#####Modelling


clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_

print(labels)



df["Clus_km"] = labels
df.head(5)


#We can easily check the centroid values by averaging the features in each cluster.
#df = df.apply(pd.to_numeric, errors='coerce')
#
#df.groupby('Clus_km').mean()

df=df.drop(df.index[[0]])


area = np.pi * ( X[:, 1])**2  
plt.scatter(Clus_dataSet[:, 1], Clus_dataSet[:, 3],marker='.')
plt.show()