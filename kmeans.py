import numpy as np
import random as rd



# k-means algorithm over high-dimensional data
class KMeans:

	def __init__(self, n_clusters=3, max_iter=300):
		self.n_clusters = n_clusters
		self.max_iter = max_iter

	def distance(self, point, data):
		"""
		L2 norm distance between point & data.
		Point has dimensions (m,d), data has dimensions (n,m,d), and output will be of size (n,).
		"""
		return np.linalg.norm(point - data, ord=2, axis=(1,2))


	def get_random_centroids(self, data):
	
		#return random samples from the dataset
		cent = (data.sample(n = k))
		return cent


	def fit(self, X_train):

		# Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
		# then the rest are initialized w/ probabilities proportional to their distances to the first
		# Pick a random point from train data for first centroid
		self.centroids = [rd.choice(X_train)]
		for _ in range(self.n_clusters-1):
			# Calculate distances from points to the centroids
			dists = np.sum([self.distance(centroid, X_train) for centroid in self.centroids], axis=0)
			# Normalize the distances
			dists /= np.sum(dists)
			# Choose remaining points based on their distances
			new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
			self.centroids += [X_train[new_centroid_idx]]
		
		# Iterate, adjusting centroids until converged or until passed max_iter
		iteration = 0
		prev_centroids = None
		while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
			# Sort each datapoint, assigning to nearest centroid
			sorted_points = [[] for _ in range(self.n_clusters)]
			for x in X_train:
				dists = self.distance(x, self.centroids)
				centroid_idx = np.argmin(dists)
				sorted_points[centroid_idx].append(x)
			# Push current centroids to previous, reassign centroids as mean of the points belonging to them
			prev_centroids = self.centroids
			self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
			for i, centroid in enumerate(self.centroids):
				if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
					self.centroids[i] = prev_centroids[i]
			iteration += 1

	def evaluate(self, X):
		centroids = []
		centroid_idxs = []
		for x in X:
			dists = self.distance(x, self.centroids)
			centroid_idx = np.argmin(dists)
			centroids.append(self.centroids[centroid_idx])
			centroid_idxs.append(centroid_idx)
		return np.array(centroids), np.array(centroid_idxs)
		