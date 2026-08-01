import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

labels = gmm.predict(X)

print("Means predicted by GMM:")
print(gmm.means_)

print("Covariances predicted by GMM:")
print(gmm.covariances_)

print("First 10 cluster predictions:")
print(labels[:10])
