import numpy as np
import pickle

import song_history_cluster as ind_cluster
import MFCC_cluster as tm_cl

#Calculates the sum of distances from a song to the centroids
def calculate_cluster_dist(song, means, clusterings, mfcc_dists, weights):
    dist = 0
    weighted_song = song[0:7] * weights[0:7]
    dists = np.zeros(means.shape[0])#min
    for cluster in range(means.shape[0]):
        dist += np.dot(weighted_song, means[cluster].T)
        a_dist = np.dot(weighted_song, means[cluster].T)#min
        avg_mfcc_dist = 0
        n = 0
        for id in range(len(clusterings)):
            if clusterings[id] == cluster:
                n += 1
                avg_mfcc_dist += mfcc_dists[cluster]
        dists[cluster] = a_dist + avg_mfcc_dist #min
        avg_mfcc_dist /= n
        dist += avg_mfcc_dist

    return np.amin(dists)

#Calculates GCD of two numbers using the Euclidean Algorithm.
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

#Finds the LCM of two numbers
def lcm(nums):
    gcd = euclid(nums[0], nums[1])
    prod = nums[0] * nums[1]
    lcm = prod/gcd
    for num in nums[2:]:
        gcd = euclid(lcm, num)
        prod = lcm * num
        lcm = prod/gcd
    return int(lcm)

#Generates a single history by weighting individual histories by their length
#and joining them together. In future iterations this should be replaced by
#weighted k-means.
def generate_full_history(histories, histories_tms):
    lengths = np.zeros((len(histories))).astype(int)
    for user in range(len(histories)):
        lengths[user] = histories[user].shape[0]
    least_multiple = lcm(lengths)
    multiples = (least_multiple / lengths).astype(int)
    full_history = []
    full_tms = []
    copies = {}
    index = 0
    for user in range(len(histories)):
        length = len(histories[user])

        for multiple in range(multiples[user]):
            full_history.extend(histories[user])
            full_tms.extend(histories_tms[user])

        for i in range(length):
            copy_array = []
            for j in range(multiples[user]):
                copy_array.append(index + i + (j * length))
            copies[index + i] = copy_array
        index += length * multiples[user]

    full_history = np.array(full_history)
    return full_history, full_tms, copies

#Finds the n songs from songs that are closest to histories and returns their indicies.
#First uses centroid distance as a heuristic to restrict the search space, then uses
#the full FSS calculation.
def find_nearest(songs, histories, songs_tms, histories_tms, n, weights):
    #history, tms, copies = generate_full_history(histories, histories_tms)
    history = histories[0]
    tms = histories_tms[0]
    copies = {}
    cs, mus, cluster_mfcc_dists, cluster_tag_dists = ind_cluster.cluster_history(history, tms, weights, len(history), copies)

    mfcc_dists = weights[8] * ind_cluster.custom_MFCC_dists(songs_tms, tms, copies, False)
    #tag_dists = weights[7] * ind_cluster.custom_tag_differences(songs, history)
    cluster_dists = np.zeros(songs.shape[0])
    for song in range(songs.shape[0]):
        cluster_dists[song] = calculate_cluster_dist(songs[song], mus, cs, mfcc_dists[song], weights)
    pickle.dump(cluster_dists, open('centroid_cluster_dists.pickle', 'wb'))
    scope = int(round(1.5 * n))
    arg_approx_smallest = np.argpartition(cluster_dists, scope)[:scope]

    heuristic_nearest = songs[arg_approx_smallest]
    heuristic_tms = []
    for index in arg_approx_smallest:
        heuristic_tms.append(songs_tms[index])
    mfcc_dists = weights[8] * ind_cluster.custom_MFCC_dists(heuristic_tms, tms, copies, False)#Change back to true
    #tag_dists = weights[7] * ind_cluster.custom_tag_differences(heuristic_nearest, history)
    cluster_dists = np.zeros(scope)
    for song in range(scope):
        cluster_dists[song] = calculate_cluster_dist(heuristic_nearest[song, :], mus, cs, mfcc_dists[song], weights)
    pickle.dump(cluster_dists, open('FSS_cluster_dists.pickle', 'wb'))
    arg_smallest = np.argpartition(cluster_dists, n)[:n]

    return arg_approx_smallest[arg_smallest]

def evaluate(playlist, histories):
    score = 0
    for history in histories:
        tag_matrix = ind_cluster.custom_tag_differences(playlist[:,7], history[:,7])
        score += np.sum(tag_matrix)
    return score

#Parameters
n = 8
weights = [.02, 3, 2.5, 1, 0.01, 2, .14, 0, 1/100000]
#weights = [.02, 1, 3, 2.5, 0.01, 3, .2, 1, 1/100000]
#weights = [0.1,0,0,0,0,0,0,1,0]
num_histories = 1
indicies = [2]

songs, song_tms, titles = ind_cluster.load_data('songs.npy', 0)
histories = [None] * num_histories
histories_tms = [None] * num_histories
for i in range(num_histories):
    histories[i], histories_tms[i], _ = ind_cluster.load_data('history' + str(indicies[i]) + '.npy', 0)
playlist_indicies = find_nearest(songs, histories, song_tms, histories_tms, n, weights)

for index in playlist_indicies:
    print(titles[index])