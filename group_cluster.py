import numpy as np
import song_history_cluster as ind_cluster
import MFCC_cluster as tm_cl

def reconstruct(data):
    tms = [None] * len(data)
    for i in range(0, num_songs_to_cluster):
        tms[i] = tm_cl.reconstruct(data[i][8:])
    return tms

#Calculates MFCC distance matrix
def calculate_all_distances(songs, playlist, weight, FSS):
    songs_tms = reconstruct(songs)
    playlist_tms = reconstruct(playlist)
    mfcc_diffs = weight*ind_clusters.custom_MFCC_dists(songs_tms, playlist_tms, FSS)
    return mfcc_diffs

#Note that MFCC_dists should be the single row of the matrix corresponding to song.
#Returns sum of distances to all clusters
def calculate_cluster_dist(song, means, clusterings, mfcc_dists, weights):
    dist = 0
    weighted_song = song[0:7] * weights[0:7]
    for cluster in range(means.shape[0]):
        dist += np.dot(weighted_song, means[cluster].T)
        avg_mfcc_dist = 0
        n = 0
        for id in range(clusterings):
            if clusterings[id] == cluster:
                n += 1
                avg_mfcc_dist += mfcc_dists[id]
        avg_mfcc_dist /= n
        dist += avg_mfcc_dist

    return dist

#Euclidean algorithm, returns gcd
def euclid(num1, num2):
    if num1 == 0:
        return num2
    if num2 == 0:
        return num1
    if num1 >= num2:
        r2 = num1 % num2
        r1 = num1
    else:
        r2 = num2 % num1
        r1 = num2
    return euclid(r1, r2)

def lcm(nums):
    gcd = euclid(nums[0], nums[1])
    prod = nums[0] * nums[1]
    lcm = prod/gcd
    for num in nums[2:]:
        gcd = euclid(lcm, num)
        prod = lcm * num
        lcm = prod/gcd
    return lcm

def generate_full_history(histories):
    lengths = np.zeros((1, len(histories)))
    for user in range(len(histories)):
        lengths[user] = histories[user].shape[0]
    lcm = lcm(lengths)
    multiples = lcm / lengths
    full_history = []
    for user in range(len(histories)):
        for _ in range(multiples[user]):
            full_history = np.concatenate((full_history, histories[user]), axis=0)
    return full_history

def find_nearest(songs, histories, n, weights):
    history = generate_full_history(histories)
    tms = reconstruct(history)
    cs, mus, cluster_mfcc_dists, cluster_tag_dists = ind_cluster.cluster_history(history, tms)

    mfcc_dists = calculate_all_distances(songs, history, weights[8], False)
    cluster_dists = np.zeros((1, songs.shape[0]))
    for song in range(songs.shape[0]):
        cluster_dists[song] = calculate_cluster_dist(songs[song, :], mus, cs, mfcc_dists, weights)
    scope = int(round(1.5 * n))
    arg_approx_smallest = np.argpartition(cluster_dists, scope)[:scope]

    heuristic_nearest = songs[arg_approx_smallest]
    mfcc_dists = calculate_all_distances(heuristic_nearest, history, weights[8], True)
    cluster_dists = np.zeros((1, scope))
    for song in range(scope):
        cluster_dists[song] = calculate_cluster_dist(heuristic_nearest[song, :], mus, cs, mfcc_dists, weights)
    arg_smallest = np.argpartition(cluster_dists, n)[:n]
    return songs[arg_smallest]