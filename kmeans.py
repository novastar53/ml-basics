import logging
from pprint import pformat

from sklearn.cluster import KMeans
from sklearn.metrics import (
	silhouette_score, 
	calinski_harabasz_score, 
	davies_bouldin_score,
	adjusted_rand_score,
)
from sklearn.datasets import make_blobs
import numpy as np 

import matplotlib.pyplot as plt 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():

	# Let's use some simple dummy data that should easily fit into 2 clusters
	true_centers = np.array([[-7, -7],[4, 1.5],[-3, 8]])
	data, true_labels = make_blobs(n_samples=30, centers=3, cluster_std=1.5, random_state=42)

	# Start with basic Kmeans.
	kmeans = KMeans(n_clusters=3, tol=1e-6)
	# Do a single fit iteration
	kmeans.fit(data)

	# Obtain the predicted cluster labels
	pred_labels = kmeans.labels_

	# Let's print out the cluster labels and see if they make sense
	logger.info("Cluster labels:")
	logger.info(f"Predicted: {pred_labels}")
	logger.info(f"Ground Truth: {true_labels}")

	# Let's analyze these results

	# How many iterations did the algorithm perform? 
	logger.info("---")
	logger.info(f"Number of iterations: {kmeans.n_iter_}")
	logger.info(f"Max iterations: {kmeans.max_iter}")

	# How much is the tolerance? (Change in the sum of squared distances)
	logger.info(f"Tolerance: {kmeans.tol}")

	# Can we use some metrics to tell how well we did? 

	# Let's first look at the Silhouette score. This indicates how well-defined the clusters are and how well
	# the clusters are separated from each other
	silhouette_avg = silhouette_score(data, kmeans.labels_)
	logger.info(f"Silhouette score [-1,1] (higher is better) {silhouette_avg}")

	# Now let's look at the Calinski-Harabasz Index (Variance Ratio Criterion)
	# This metric also gives an indication of cluster separation and compactness
	ch = calinski_harabasz_score(data, kmeans.labels_)
	logger.info(f"Calinski-Harabasz Index (Variance Ratio Criterion) (Higher is better) {ch}")

	# Another metric worth looking at is the Davies-Bouldin Index. This is a bit more complicated,
	# but it essentially measures the same thing - cluster separation and compactness. Lower is better
	db = davies_bouldin_score(data, kmeans.labels_)
	logger.info(f"Davies-Bouldin index (lower is better) {db}")

	# If you have the ground-truth labels, you could use a similarity metric between the predicted and true 
	# labels to calculate some kind of similarity metric. For example, the Adjusted Rand Index (ARI).
	ari = adjusted_rand_score(true_labels, kmeans.labels_)
	logger.info(f"Adjusted Rand Score [-1,1] (Higher is better) {ari}")

	# Where are the cluster centers located? 
	cluster_centers = kmeans.cluster_centers_
	logger.info(f"Cluster centers: {cluster_centers}")

	# Let's plot the cluster centers and data points on a scatter plot
	plt.scatter(data[:, 0], data[:, 1], c=pred_labels, marker='o', label='Data Points')
	plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, c='red', marker='x', label='Predicted Centroids')
	plt.scatter(true_centers[:, 0], true_centers[:, 1], s=200, c='green', marker='x', label='True Centroids')
	plt.title('K-means Clustering Results')
	plt.legend()
	plt.show()


if __name__ == "__main__":
	main()