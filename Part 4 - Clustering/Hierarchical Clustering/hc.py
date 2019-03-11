# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 3:].values

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('dendrogram')
plt.xlabel('customers')
plt.ylabel('euclidean distance')
plt.show()

# Fitting hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=30, c='red', label='careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=30, c='blue', label='standart')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=30, c='green', label='target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=30, c='cyan', label='careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=30, c='magenta', label='sensible')
plt.title('cluster of clients')
plt.xlabel('annual income (k$)')
plt.ylabel('spending score (1-100)')
plt.legend()
plt.show()