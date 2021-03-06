import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
import scipy.special as sp
import sys
import math

# Reconstructs the timbre matrix from a flattened vectors, which is 
# how the data is read in
def reconstruct(tm_flat):
	return(np.reshape(tm_flat, (-1, int(tm_flat.shape[0]/12))))

# Collapsing k-means along with a condition to make sure every cluster
# has size of at least 2, which is what is required to fit a diagonal
# covariance matrix.
def cluster_song(tm, threshold = 1000, min_clusters = 4, max_it = 100):
	kmeans = None
	k = 15
	first = True

	converged = False
	it = 0
	distortion = 0
	while (k > min_clusters and it < max_it):
		it += 1
		if (first):
			first = False
			kmeans = KMeans(n_clusters=k, max_iter=1).fit(tm)
			centroids = kmeans.cluster_centers_
			continue; 

		kmeans = KMeans(n_clusters=k, init=centroids, max_iter=1).fit(tm)
		# Collapse centroids that are too close together
		new_centroids = collapse_centroids(kmeans)
		new_distortion = 0
		for i in range(kmeans.labels_.shape[0]):
			centroid_assignment = centroids[kmeans.labels_[i]]
			error = tm[i,:] - centroid_assignment
			new_distortion += np.dot(error.T, error)

		# Check for convergence if we haven't merged two clusters on this iteration
		if (new_centroids.shape == centroids.shape and np.linalg.norm(new_distortion - distortion) < threshold):
			centroids = new_centroids
			converged = True
			break
		else:
			k = len(new_centroids)
			centroids = new_centroids
		distortion = new_distortion

	if it == max_it:
		kmeans = KMeans(n_clusters=k).fit(tm)
		converged = True

	if not converged:
		kmeans = KMeans(n_clusters=k, init=centroids).fit(tm)
		centroids = kmeans.cluster_centers_

	# Find clusters of size 1
	not_usable = set()
	usable = set()
	keep = []
	labels = list(kmeans.labels_)
	counts = {}
	new_centroids = []

	for i in labels:
		if i not in counts:
			counts[i] = 0
		counts[i] += 1
	for count in counts:
		if counts[count] < 2:
			not_usable.add(count)
		else:
			usable.add(count)
			new_centroids.append(centroids[count])

	# Keep clusters that have size 2 or greater
	for i in range(len(labels)):
		if labels[i] in usable:
			keep.append(i)
			new_list = list(not_usable)
			new_list.append(int(labels[i]))
			new_list.sort()
			ind = new_list.index(labels[i])
			labels[i] -= ind
	keep = np.array(keep)

	# Drop vectors belonging to size 1 clusters
	return np.array(new_centroids), np.array(labels)[keep], tm[keep,:]


# Concatenate window_length MFCC vectors together along with deltas
# to make cloud vectors
def generate_cloud(timbre_matrix, window_length):
	length = timbre_matrix.shape[1]
	vectors = np.zeros((length - window_length, (window_length+1) * 12))
	for index in range(length - window_length):
		vector = timbre_matrix[:,index:index + window_length].flatten()
		deltas = timbre_matrix[:,index+window_length] - timbre_matrix[:,index]
		vectors[index,:] = np.concatenate((vector,deltas))
	return(vectors)

# Check centroids and collapse together if any are closer than threshold
# Threshold was tuned by inspecting output cluster numbers
def collapse_centroids(kmeans):
	threshold = 200

	centroids = kmeans.cluster_centers_
	labels = kmeans.labels_
	counts = {}
	for i in labels:
		if i not in counts:
			counts[i] = 0
		counts[i] += 1

	# Create matrix of distances between centroids
	shape = centroids.shape[0]
	distances = np.zeros((shape,shape))
	for i in range(shape):
		distances[i][i] = float("inf")
		for j in range(i+1,shape):
			distances[i][j] = np.linalg.norm(centroids[j]-centroids[i])
			distances[j][i] = distances[i][j]

	# Loop through and collapse if any two are too close. Multiple
	# collapses of the same cluster are disallowed
	new_centroids = []
	already_combined = set()
	for i in range(shape):
		min_dist = min(distances[i]);
		min_index = np.nonzero(distances[i] == min_dist)[0][0]
		if ((min_dist < threshold) and i not in already_combined):
			new_centroid =(counts[i]*centroids[i] + counts[min_index]*centroids[min_index])/(counts[i]+counts[min_index])
			new_centroids.append(new_centroid)
			already_combined.add(min_index)
			for j in range(shape):
				distances[j,i] = float('inf')
				distances[min_index,j] = float('inf')

		elif (i not in already_combined):
			new_centroids.append(centroids[i])

	return(np.array(new_centroids))

#M-step, but just with the priors.
def maximize_priors(cloud, means, covariances):
	priors = np.zeros((1, means.shape[1]))
	priors.fill(1/means.shape[1])
	delta = 10
	while(delta > 0.1):
		posteriors = calculate_posteriors(cloud, means, covariances, priors)
		old_priors = priors
		priors = np.sum(posteriors, axis=0, keepdims=True)
		priors /= cloud.shape[0]
		delta = np.linalg.norm(priors-old_priors)
	return priors

#E-step: Calculate soft guesses for latent variables.
def calculate_posteriors(data, means, covariances, priors):
	posteriors = np.zeros((data.shape[0],means.shape[1]))
	for i in range(data.shape[0]):
		x = data[i,:]
		for j in range(means.shape[1]):
			mean = means[:,j]
			covariance = covariances[j,:,:]
			prior = priors[0,j] + 1e-100
			log_like = multivariate_normal.logpdf(x, mean, covariance)
			posterior = np.log(prior) + log_like
			posteriors[i,j] = posterior
		posteriors[i,:] -= sp.logsumexp(posteriors[i,:])
	return np.exp(posteriors)

#Calculates the fast spectral distance between two GMM models by using the 
#cross similarity metric, normalized by the self similarity, as described in one of
#the papers.
def calculate_distance(priors1, priors2, means1, means2, covariances1, covariances2):
	L_AA = likelihood(priors1, priors1, means1, means1, covariances1)
	L_BB = likelihood(priors2, priors2, means2, means2, covariances2)
	self_similarity = L_AA + L_BB
	L_AB = likelihood(priors1, priors2, means1, means2, covariances2)
	L_BA = likelihood(priors2, priors1, means2, means1, covariances1)
	cross_similarity = L_AB + L_BA
	distance = cross_similarity - self_similarity
	return distance

#Calculates the likelihood of a set of centroids (means1, priors1) occuring given
#the GMM defined by means2, covariances, priors2.
def likelihood(priors1, priors2, means1, means2, covariances):
	likelihood = 0
	for i in range(priors1.shape[1]):
		weight_a = priors1[0,i] + 1e-100
		likelihoods_b = np.zeros((1, priors2.shape[1]))
		for j in range(priors2.shape[1]):
			weight_b = priors2[0,j] + 1e-100
			log_prob = multivariate_normal.logpdf(means1[:,i], mean=means2[:,j], cov=covariances[j,:,:])
			likelihoods_b[0,j] = np.log(weight_b) + log_prob

		likelihood_a = sp.logsumexp(a=likelihoods_b)
		likelihood += weight_a * likelihood_a
	return likelihood

#Calculates the total normalized centroid distance between two GMMs
def total_cen_distance(means1, means2):
	return 2*centroid_distance(means1, means2) - centroid_distance(means1, means1) - centroid_distance(means2, means2)

#Calculates the normalized sum of centroid distances
def centroid_distance(means1, means2):
	dist = 0
	for i in range(means1.shape[1]):
		for j in range(means2.shape[1]):
			diff = means1[:,i] - means2[:,j]
			dist += np.dot(diff, diff.T)
	dist /= (means1.shape[1] * means2.shape[1])
	return dist