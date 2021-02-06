import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets._samples_generator import make_blobs

x,y = make_blobs(n_samples=100, centers=5, random_state=0, cluster_std=2)
plt.scatter(x[:,0],x[:,1],s=75)

plt.show()

estimators = KMeans(n_clusters=6)
estimators.fit(x)
y_kmeans = estimators.predict(x)

plt.scatter(x[:,0],x[:,1],c=y_kmeans, s=75, cmap='rainbow')
plt.show()
