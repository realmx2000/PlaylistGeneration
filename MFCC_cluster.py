import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

def reconstruct(tm_flat):
	return(np.reshape(tm_flat, (-1, int(tm_flat.shape[0]/12))))

def cluster_song(tm, threshhold = 10, min_clusters = 4, max_it = 100):
	# TUNE THESE
	#.0001 increases the spread of numbers of clusters, but not the mean number.
	threshold = 10000
	#Lots of songs naturally go to 1 cluster essentailly regardless of threshhold without this.
	min_clusters = 1

	kmeans = None
	k = 30
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
		new_centroids = collapse_centroids(kmeans)
		new_distortion = 0
		for i in range(kmeans.labels_.shape[0]):
			centroid_assignment = centroids[kmeans.labels_[i]]
			error = tm[i,:] - centroid_assignment
			#print(error.shape)
			new_distortion += np.dot(error.T, error)
		#print(distortion)
		if(new_centroids.shape == centroids.shape):
			print (new_distortion - distortion)


		if (new_centroids.shape == centroids.shape and np.linalg.norm(new_distortion - distortion) < threshold):
			#print("TRUE")
			centroids = new_centroids
			converged = True
			break
		else:
			k = len(new_centroids)
			centroids = new_centroids
		distortion = new_distortion
	if it == max_it:
		print("Did this")
		kmeans = KMeans(n_clusters=k).fit(tm)
		converged = True


	if not converged:
		kmeans = KMeans(n_clusters=k, init=centroids).fit(tm)
		centroids = kmeans.cluster_centers_

	return centroids, kmeans.labels_

def generate_cloud(timbre_matrix, window_length):
	length = timbre_matrix.shape[1]
	vectors = np.zeros((length - window_length, (window_length+1) * 12))
	for index in range(length - window_length):
		vector = timbre_matrix[:,index:index + window_length].flatten()
		deltas = timbre_matrix[:,index+window_length] - timbre_matrix[:,index]
		vectors[index,:] = np.concatenate((vector,deltas))
		prev_vector = vector
	return(vectors)


def collapse_centroids(kmeans):
	# TUNE THIS
	threshold = 200

	centroids = kmeans.cluster_centers_
	labels = kmeans.labels_
	counts = {}
	for i in labels:
		if i not in counts:
			counts[i] = 0
		counts[i] += 1

	shape = centroids.shape[0]
	distances = np.zeros((shape,shape))
	for i in range(shape):
		distances[i][i] = float("inf")
		for j in range(i+1,shape):
			distances[i][j] = np.linalg.norm(centroids[j]-centroids[i])
			distances[j][i] = distances[i][j]

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
	old_priors = np.zeros((1, means.shape[0]))
	while(np.linalg.norm(priors-old_priors) > 0.1):
		posteriors = calculate_posteriors(cloud, means, covariances, priors)
		old_priors = priors
		priors = np.sum(posteriors, axis=0, keepdims=True)
		priors /= cloud.shape[0]
	return priors

#E-step: Calculate soft guesses for latent variables.
def calculate_posteriors(data, means, covariances, priors):
	posteriors = np.zeros((data.shape[0],means.shape[1]))
	for i in range(data.shape[0]):
		marginal = 0
		x = data[i,:]
		for j in range(means.shape[1]):
			mean = means[:,j]
			covariance = covariances[j,:,:]
			prior = priors[0,j]
			likelihood = multivariate_normal.pdf(x, mean, covariance)
			posterior = likelihood * prior
			marginal += posterior
			posteriors[i,j] = posterior
		posteriors[i, :] /= marginal

	return posteriors

#Means and covariances can be calculated with np.mean and np.cov, respectively.

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
		weight_a = priors1[0,i]
		likelihood_a = 0
		for j in range(priors2.shape[1]):
			weight_b = priors2[0,j]
			likelihood_b = multivariate_normal.pdf(means1[:,i], mean=means2[:,j], cov=covariances[j,:,:])
			likelihood_a += weight_b * likelihood_b
		likelihood_a = np.log(likelihood_a)
		likelihood += weight_a * likelihood_a
	return likelihood
