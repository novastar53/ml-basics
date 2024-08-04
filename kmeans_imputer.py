# kmeans_imputer.py

# Missing data is a common problem that all data scientists encounter
# with real-world data. One approach is the use K-Means to cluster 
# the data, and then use the centroids of the computed clusters to
# impute values for the missing dimensions. 

# %%
import logging
from pprint import pformat


import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Let's start by generating some sample data that is distributed 
# in 3 clusters
data, true_labels = make_blobs(n_samples=100, centers=3, cluster_std=1.0, random_state=42)

# Now let's randomly remove 10% of the data 
mask = np.random.rand(*data.shape) < 0.1
data[mask] = np.nan

# Let's do an initial round of imputation using the mean
imputer = SimpleImputer(strategy="mean")
data_imputed = imputer.fit_transform(data)

# Now let's cluster the data using K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_imputed)

# How many iterations were performed? 
logger.info(f"Number of iterations: {kmeans.n_iter_}")

# Predict the cluster centroids and use them to perform 
# the final imputation
cluster_centers = kmeans.cluster_centers_
for i in range(data.shape[0]):
    if np.any(np.isnan(data[i])):
        cluster = kmeans.predict(data_imputed[i].reshape(1,-1))
        data_imputed[i, np.isnan(data[i])] = kmeans.cluster_centers_[cluster, np.isnan(data[i])]


#logger.info("Original data with missing values")
#logger.info(pformat(data))
#logger.info("Imputed data:")
#logger.info(pformat(data_imputed))

# Now let's make some plots to see how we did

# Start by plotting the original data
plt.scatter(data[:, 0], data[:, 1], marker='o', color='black', label='Data Points', s=1) 
plt.scatter(data_imputed[:, 0], data_imputed[:, 1], marker='_', s=2, color='blue', label='Data Points')
plt.show()
# %%
