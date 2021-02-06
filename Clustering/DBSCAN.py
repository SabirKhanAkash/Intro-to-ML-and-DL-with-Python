import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import datasets
import numpy as np


x,y = datasets.make_moons(n_samples=2000, noise=0.05)

x1 = x[:,0]
x2 = x[:,1]

plt.title("This is the dataset we want to classify with DBSCAN:\n")
plt.scatter(x1,x2,s=5,color='purple')
plt.show()


dbscan = DBSCAN(eps=0.1)
dbscan.fit(x)
y_pred = dbscan.labels_.astype(np.int)

colors = np.array(['#ff0345', '#70ff09'])

plt.title("These are the clusters with DBSCAN:\n")
plt.scatter(x1,x2,s=5,color=colors[y_pred])
plt.show()


kmeans = KMeans(n_clusters=2)
kmeans.fit(x)
y_pred = kmeans.labels_.astype(np.int)

colors = np.array(['#ff0345', '#70ff09'])

plt.title("These are the clusters with K-Means:\n")
plt.scatter(x1,x2,s=5,color=colors[y_pred])
plt.show()
