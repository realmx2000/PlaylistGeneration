import numpy as np
from numpy import newaxis
from sklearn.cluster import KMeans

import MFCC_cluster as tm_cl


# Accepts mfcc matrices (reconstructed, not cloud)
def MFCC_dists(mfccs):
    num_songs = len(mfccs)
    dist_matrix = np.zeros((num_songs, num_songs))

    centers = [None] * num_songs
    labels = [None] * num_songs
    cloud = [None] * num_songs
    for i in range(num_songs):
        cloud[i] = tm_cl.generate_cloud(mfccs[i].astype(float), 5)
        centers[i], labels[i], cloud[i] = tm_cl.cluster_song(cloud[i])
        cloud[i] = cloud[i].T
        centers[i] = centers[i].T

    for i in range(num_songs):
        num_clusters_i = centers[i].shape[1]
        covs1 = [None] * num_clusters_i
        for l in range(num_clusters_i):
            covs1[l] = diag_cov(cloud[i][:,labels[i] == l])
        covs1 = np.array(covs1)

        counts1 = {}
        for w in labels[i]:
            if w not in counts1:
                counts1[w] = 0
            counts1[w] += 1

        for j in range(i+1, num_songs):

            counts2 = {}
            for w in labels[j]:
                if w not in counts2:
                    counts2[w] = 0
                counts2[w] += 1

            num_clusters_j = centers[j].shape[1]
            covs2 = [None] * num_clusters_j
            for m in range(num_clusters_j):
                covs2[m] = diag_cov(cloud[j][:,labels[j] == m])
            covs2 = np.array(covs2)

            priors1 = tm_cl.maximize_priors(cloud[i].T, centers[i], covs1)
            priors2 = tm_cl.maximize_priors(cloud[j].T, centers[j], covs2)

            dist_matrix[i,j] = tm_cl.calculate_distance(priors1, priors2, centers[i], centers[j], covs1, covs2)
            dist_matrix[j,i] = dist_matrix[i,j]
    return(dist_matrix)

def diag_cov(data):
    variances = np.var(data, axis=1)
    return np.diag(variances)

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
weight = [np.array([1,1,1,1,1,1,1,1,1]),np.array([1,1,1,1,1/200,1,1/10,1,.01]),np.array([1,1,1,1,1/200,1,1/10,1,1]),np.array([0,0,0,0,0,0,0,0,1]),np.array([1,1,1,1,1/200,1,1/10,0,1]),np.array([1,0,0,0,1/200,1,1/10,1,1]),np.array([1,0,0,0,1/200,1,1/10,1,1/100]),np.array([1,1/10,1/10,1/10,1/200,1,1/10,1,2]),np.array([1,1,1,1,1,1,1,1,5]),np.array([1,1,0,1,.01,.4,.3,.3,.1])]
for weights in weight:

    data = np.load('TestCase.npy')
    new_data = []
    for i in data:
        new_data.append(i[2:])
    new_data = np.array(new_data)
    data = new_data

    num_songs = data.shape[0]
    num_songs_to_cluster = 14
    tms = [None] * num_songs_to_cluster
    converted = [None] * num_songs_to_cluster
    for i in range(0, num_songs_to_cluster):
        tms[i] = tm_cl.reconstruct(data[i][8:])
        converted[i]= data[i][0:3].astype(np.float64)
        converted[i] = np.concatenate((converted[i], data[i][4:8].astype(np.float64)))
        converted[i] = np.concatenate((converted[i], [set(data[i][3].decode('UTF-8').split('\t'))]))
    converted = np.array(converted)

    tag_diffs = weights[7]*tag_differences(converted[:,7])
    print("Data processed, tag matrix calculated")

    mfcc_diffs = weights[8]*MFCC_dists(tms)
    print("MFCC matrix calculated")
    print(converted[:,0:7])
    print(mfcc_diffs)
    converted = np.dot(converted[:,0:7],np.diag(weights[0:7]))
    cs, mus, cluster_mfcc_dists, cluster_tag_dists = k_means2(converted[0:num_songs_to_cluster], 3, 20, mfcc_diffs, tag_diffs)
    print(cs)


