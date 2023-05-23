import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate random data for clustering
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Visualize the data
plt.scatter(X[:, 0], X[:, 1], s=50);

# Create K-means clustering model with 4 clusters
kmeans = KMeans(n_clusters=4)

# Fit the data to the model
kmeans.fit(X)

# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualize the clusters and centroids
colors = ['r', 'g', 'b', 'y']
for i in range(len(X)):
    plt.scatter(X[i][0], X[i][1], c=colors[labels[i]], s=50)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
plt.show()
