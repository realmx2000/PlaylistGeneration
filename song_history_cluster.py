import numpy as np
from numpy import newaxis
from sklearn.cluster import KMeans
import pickle

import MFCC_cluster as tm_cl

#Processes the MFCC matrix into a GMM
def get_mfcc_params(mfccs):
    num_songs = len(mfccs)
    centers = [None] * num_songs
    labels = [None] * num_songs
    cloud = [None] * num_songs

    for i in range(num_songs):
        cloud[i] = tm_cl.generate_cloud(mfccs[i].astype(float), 5)
        centers[i], labels[i], cloud[i] = tm_cl.cluster_song(cloud[i])
        cloud[i] = cloud[i].T
        centers[i] = centers[i].T

    return centers, labels, cloud

#Gets the diagonal covariances for each Gaussian in the GMM.
def get_covariances(centers, labels, cloud, i):
    num_clusters_i = centers[i].shape[1]
    covs = [None] * num_clusters_i
    for l in range(num_clusters_i):
        covs[l] = diag_cov(cloud[i][:, labels[i] == l])
    covs = np.array(covs)
    return covs

#Calculate distance matrix of MFCC distances, using FSS or Centroid Distance.
def custom_MFCC_dists(mfccs1, mfccs2, FSS):
    num_songs1 = len(mfccs1)
    num_songs2 = len(mfccs2)
    dist_matrix = np.zeros((num_songs1, num_songs2))
    centers1, labels1, cloud1 = get_mfcc_params(mfccs1)
    centers2, labels2, cloud2 = get_mfcc_params(mfccs2)

    for i in range(len(mfccs1)):
        print("Distance from %d" % i)
        if FSS:
            covs1 = get_covariances(centers1, labels1, cloud1, i)

        for j in range(len(mfccs2)):
            print("To %d" % j)

            if FSS:
                covs2 = get_covariances(centers2, labels2, cloud2, j)
                priors1 = tm_cl.maximize_priors(cloud1[i].T, centers1[i], covs1)
                priors2 = tm_cl.maximize_priors(cloud2[j].T, centers2[j], covs2)
                dist_matrix[i,j] = tm_cl.calculate_distance(priors1, priors2, centers1[i], centers2[j], covs1, covs2)
            else:
                dist_matrix[i,j] = tm_cl.total_cen_distance(centers1[i], centers2[j])
    return dist_matrix

#Calculates MFCC Distances between a set of MFCC's and itself using FSS or Centroid Distance.
def MFCC_dists(mfccs, FSS):
    num_songs = len(mfccs)
    dist_matrix = np.zeros((num_songs, num_songs))

    centers, labels, cloud = get_mfcc_params(mfccs)

    for i in range(num_songs):
        print("Distance from %d" % i)
        if FSS:
            covs1 = get_covariances(centers, labels, cloud, i)

        for j in range(i + 1, num_songs):
            print("To %d" % j)

            if FSS:
                covs2 = get_covariances(centers, labels, cloud, j)
                priors1 = tm_cl.maximize_priors(cloud[i].T, centers[i], covs1)
                priors2 = tm_cl.maximize_priors(cloud[j].T, centers[j], covs2)
                dist_matrix[i,j] = tm_cl.calculate_distance(priors1, priors2, centers[i], centers[j], covs1, covs2)
            else:
                dist_matrix[i,j] = tm_cl.total_cen_distance(centers[i], centers[j])
            dist_matrix[j,i] = dist_matrix[i,j]
    return dist_matrix

#Calculates a diagonal covariance matrix for the data
def diag_cov(data):
    variances = np.var(data, axis=1)
    return np.diag(variances)

#Calculates the IOU of the tags
def tag_differences(tag_lists):
    num_songs = len(tag_lists)
    tag_matrix = np.zeros((num_songs, num_songs))
    for i in range(num_songs):
        for j in range(i+1, num_songs):
            shared = len(tag_lists[i] & tag_lists[j])
            total = len(tag_lists[i] | tag_lists[j])
            tag_matrix[i,j] = 1-(float(shared)/total)
            tag_matrix[j,i] = tag_matrix[i,j]
    return(tag_matrix)

# k means implementation including the MFCC distances and tags
def k_means(data, c_count, max_iter, dist_matrix, tag_matrix):
    data = data.astype(np.float64)
    indices = np.random.choice(data.shape[0], c_count, replace=False)
    # initial centers
    mus = data[indices, :]
    # initial cluster dists
    cluster_mfcc_dists = dist_matrix[indices]
    cluster_tag_dists = tag_matrix[indices]
    distortion = 0
    prev_distortion = 0
    
    for rep in range(max_iter):
        # choose centers
        diff = data[newaxis, :, :] - mus[:, newaxis, :]
        dist = np.sqrt(np.square(np.linalg.norm(diff, axis=2)) + np.square(cluster_mfcc_dists) + np.square(cluster_tag_dists))
        cs = np.argmin(dist, axis=0)

        # compute new centers
        for j in range(mus.shape[0]):
            mus[j] = np.sum(data[cs == j], axis=0) / len(data[cs == j])

        # calculate distance rows of new clusters
        for j in range(c_count):
            cluster_mfcc_dists[j] = np.sum(dist_matrix[cs == j], axis=0) / len(dist_matrix[cs == j])
            cluster_tag_dists[j] = np.sum(tag_matrix[cs == j], axis=0) / len(tag_matrix[cs == j])

        # compute distortion
        distortion = 0
        for i in range(data.shape[0]):
            error = data[i,:] - mus[cs[i]]
            distortion += error.dot(error) + np.square(cluster_mfcc_dists[cs[i],i]) + np.square(cluster_tag_dists[cs[i],i])

        # break if distortion does not change
        if rep != 0 and prev_distortion - distortion < .01:
            break
        prev_distortion = distortion

    return cs, mus, cluster_mfcc_dists, cluster_tag_dists

#Clusters a set of songs. Returns the parameters of the clustering.
def cluster_history(history, tms, weights, num_songs_to_cluster):
    #TODO: Tag Differences
    tag_diffs = weights[7] * tag_differences(history[:, 7])

    mfcc_diffs = weights[8] * MFCC_dists(tms, False) #TODO: Switch back to true
    pickle.dump(mfcc_diffs, open('distances.pickle', 'wb'))
    history = np.dot(history[:,0:7],np.diag(weights[0:7]))
    cs, mus, cluster_mfcc_dists, cluster_tag_dists = k_means(history[0:num_songs_to_cluster], 3, 20, mfcc_diffs, tag_diffs)
    return cs, mus, cluster_mfcc_dists, cluster_tag_dists

#Loads songs in and formats the data.
def load_data(name, num_songs_to_cluster):
    data = np.load(name, encoding='bytes')
    if num_songs_to_cluster == 0:
        num_songs_to_cluster = data.shape[0]
    titles = []
    new_data = []
    for i in data:
        new_data.append(i[2:])
        titles.append(i[:2])
    new_data = np.array(new_data)
    data = new_data

    tms = [None] * num_songs_to_cluster
    converted = [None] * num_songs_to_cluster
    for i in range(0, num_songs_to_cluster):
        tms[i] = tm_cl.reconstruct(data[i][8:])
        converted[i]= data[i][0:3].astype(np.float64)
        converted[i] = np.concatenate((converted[i], data[i][4:8].astype(np.float64)))
        converted[i] = np.concatenate((converted[i], [set(data[i][3].decode('UTF-8').split('\t'))]))
    converted = np.array(converted)
    return converted, tms, titles

# Script

# TUNE THIS
# Features: Tempo, Familiarity, Hotness, Danceability, Duration, Energy, Loudness, Terms, MFCC
#weight = [np.array([1,1,1,1,1,1,1,1]),np.array([1,1,1,1/200,1,1/10,1,1]),np.array([0,0,0,0,0,0,0,1]),np.array([1,1,1,1/200,1,1/10,0,1]),np.array([1,1,1,1/200,1,1/10,1,.01]),np.array([0,0,0,1/200,1,1/10,1,1]),np.array([0,0,0,1/200,1,1/10,1,1/100]),np.array([1/10,1/10,1/10,1/200,1,1/10,1,2]),np.array([1,1,1,1,1,1,1,5]),np.array([1,0,1,.01,.4,.3,.3,.1])]
#more weights = np.array([1, 0.1, 0.3, 1, 0.2, 1, 0.5, 1, 1.2]), 

"""
[0 0 0 0 0 0]
[1 1 1 0 1 1]
[0 0 1 0 0 0]
[1 1 1 1 0 1]
[0 0 0 1 0 0]
[1 1 1 0 1 1]
[0 0 0 1 1 0]
[0 0 0 0 1 0]
[1 1 1 1 0 1]
[1 1 0 1 1 1]"""

#SONGS
"""
("I Didn't Mean To", 'Casual') Rap
('Soul Deep', 'The Box Tops') Folk
('Amor De Cabaret', 'Sonora Santanera') Weird slow Spanish
('Something Girls', 'Adam Ant') Folk/Rock
('Face the Ashes', 'Gob') Rock/Metal
('The Moon And I (Ordinary Day Album Version)', 'Jeff And Sheri Easter') Country
"""
#weight = [np.array([1,1,1,1,1,1,1,1,1]),np.array([1,1,1,1,1/200,1,1/10,1,.01]),np.array([1,1,1,1,1/200,1,1/10,1,1]),np.array([0,0,0,0,0,0,0,0,1]),np.array([1,1,1,1,1/200,1,1/10,0,1]),np.array([1,0,0,0,1/200,1,1/10,1,1]),np.array([1,0,0,0,1/200,1,1/10,1,1/100]),np.array([1,1/10,1/10,1/10,1/200,1,1/10,1,2]),np.array([1,1,1,1,1,1,1,1,5]),np.array([1,1,0,1,.01,.4,.3,.3,.1])]
#weight = [np.array([1, 0, 0, 0, 0, 0, 1, 1, 0.1])]

'''
weight = [np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])]
num_songs_to_cluster=14
converted, tms, _ = load_data('TestCase.npy', num_songs_to_cluster)

for weights in weight:
    cs, mus, cluster_mfcc_dists, cluster_tag_dists = cluster_history(converted, tms, weights, num_songs_to_cluster)
'''