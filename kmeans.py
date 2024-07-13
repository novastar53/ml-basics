import logging
from pprint import pformat

from sklearn.cluster import KMeans
import numpy as np 

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Let's use some simple dummy data that should easily fit into 2 clusters
data = np.array([[1,2], [1,4], [1,0], [10,2], [10,4], [10,0]])


# Start with basic Kmeans.
kmeans = KMeans(n_clusters=2, random_state=42)
# Do a single fit iteration
kmeans.fit(data)

# Obtain the predicted cluster labels
labels = kmeans.labels_

# Let's print out the cluster labels and see if they make sense
logger.info("Cluster labels:")
logger.info(f"\n{pformat(list(zip(data,labels)))}")
