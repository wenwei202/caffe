__author__ = 'caffe'

import numpy as np
import matplotlib.pyplot as plt
from pittnuts import *
import copy
from sklearn.cluster import KMeans

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)

def get_blob_sparsity(net):
    sparsity = {}
    for blob_name in net.blobs.keys():
         sparsity[blob_name] = get_sparsity(net.blobs[blob_name].data)
    return sparsity

def rank_by_ratio(eig_values,ratio):
    assert ratio<=1 and ratio>0
    eig_values = copy.copy(eig_values)
    eig_sum = sum(eig_values)
    for i in range(1, eig_values.size):
        eig_values[i] = eig_values[i] + eig_values[i - 1]
    eig_values = eig_values / eig_sum
    # return the rank that keeps ratio information
    for i in range(0, eig_values.size):
        if eig_values[i]>=ratio:
            return i+1
    return eig_values.size

def rank_by_ratio_2(singular_values,ratio):
    assert ratio<=1 and ratio>0
    assert (singular_values>=0).all()
    singular_values = copy.copy(singular_values)
    singular_val_norm = np.linalg.norm(singular_values)
    for i in range(1, singular_values.size):
        singular_values[i] = np.sqrt(singular_values[i]**2 + singular_values[i - 1] ** 2)
    assert abs(singular_values[-1] - singular_val_norm)<0.001
    singular_values = singular_values / singular_val_norm
    # return the rank that keeps ratio information
    for i in range(0, singular_values.size):
        if singular_values[i]>=ratio:
            return i+1
    return singular_values.size

def filter_pca(filter_weights,ratio=None,rank=None):
    filter_num = filter_weights.shape[0]
    chan_num = filter_weights.shape[1]
    kernel_h = filter_weights.shape[2]
    kernel_w = filter_weights.shape[3]
    kernel_size = kernel_h * kernel_w
    # decompose the weights
    weights_pca = filter_weights.reshape((filter_num, chan_num * kernel_size)).transpose()
    #weights_mean = mean(weights_pca, axis=0)
    weights_pca, eig_vecs, eig_values = pca(weights_pca)
    if None != ratio:
        rank = rank_by_ratio(eig_values, ratio)
    #shift_vals = dot(weights_mean, eig_vecs[:, rank:])
    weights_full = weights_pca.transpose().reshape((filter_num, chan_num, kernel_h, kernel_w))
    low_rank_filters = weights_full[0:rank]
    #if rank >= filter_num - 1:
    linear_combinations = eig_vecs[:, 0:rank].reshape((filter_num, rank, 1, 1))
    return (low_rank_filters, linear_combinations, rank)
    #else:
    #    mean_compensation = dot(shift_vals,eig_vecs[:, rank:].transpose()).transpose().reshape((-1, 1))
    #    length = np.linalg.norm(mean_compensation)
    #    linear_combinations = np.concatenate((eig_vecs[:, 0:rank],mean_compensation/length),axis=1)
    #    linear_combinations = linear_combinations.reshape((filter_num, rank+1, 1, 1))
    #    # append an all-ones filter to compensate the deviation resulted from non-zero mean filters
    #    low_rank_filters = np.concatenate((low_rank_filters, np.ones((1, chan_num, kernel_h, kernel_w))*length), axis=0)
    #    return (low_rank_filters,linear_combinations,rank+1)

def fc_pca(fc_weights,ratio=None,rank=None):
    # decompose the weights
    weights_pca = fc_weights.transpose()
    weights_pca, eig_vecs, eig_values = pca(weights_pca)
    if None != ratio:
        rank = rank_by_ratio(eig_values, ratio)
    weights_full = weights_pca.transpose()
    low_rank_a = weights_full[0:rank]
    low_rank_b = eig_vecs[:, 0:rank]
    return (low_rank_a, low_rank_b, rank)

def filter_kmeans(filter_weights,rank):
    if rank==None:
        print "rank is None"
        exit()
    filter_num = filter_weights.shape[0]
    chan_num = filter_weights.shape[1]
    kernel_h = filter_weights.shape[2]
    kernel_w = filter_weights.shape[3]
    kernel_size = kernel_h * kernel_w
    # decompose the weights
    weights_kmeans = filter_weights.reshape((filter_num, chan_num * kernel_size))
    kmeans = KMeans(n_clusters=rank).fit(weights_kmeans)
    low_rank_filters = kmeans.cluster_centers_.reshape((rank, chan_num, kernel_h, kernel_w))
    cluster_idx = kmeans.predict(weights_kmeans)
    linear_combinations = np.eye(rank)[cluster_idx].reshape((filter_num, rank, 1, 1))
    return (low_rank_filters, linear_combinations, rank)

def filter_svd(filter_weights,ratio=None,rank=None):
    filter_num = filter_weights.shape[0]
    chan_num = filter_weights.shape[1]
    kernel_h = filter_weights.shape[2]
    kernel_w = filter_weights.shape[3]
    kernel_size = kernel_h * kernel_w
    # decompose the weights
    weights = filter_weights.reshape((filter_num, chan_num * kernel_size)).transpose()
    u, s, v = np.linalg.svd(weights,full_matrices=False)
    if None != ratio:
        rank = rank_by_ratio_2(s, ratio)
    sqrt_singular_val = np.sqrt(np.diag(s))
    u = dot(u,sqrt_singular_val)
    v = dot(sqrt_singular_val,v)
    weights_full = u.transpose().reshape((filter_num, chan_num, kernel_h, kernel_w))
    low_rank_filters = weights_full[0:rank]
    linear_combinations = v[0:rank,:].transpose().reshape((filter_num, rank, 1, 1))
    a = low_rank_filters.reshape((-1, chan_num * kernel_size)).transpose()
    b = linear_combinations.reshape((filter_num,-1)).transpose()
    print np.linalg.norm(subtract(weights,dot(a,b))), np.linalg.norm(s[rank:])
    return (low_rank_filters, linear_combinations, rank)

def fc_svd(fc_weights,ratio=None,rank=None):
    # decompose the weights
    weights = fc_weights.transpose()
    u, s, v = np.linalg.svd(weights,full_matrices=False)
    if None != ratio:
        rank = rank_by_ratio_2(s, ratio)
    sqrt_singular_val = np.sqrt(np.diag(s))
    u = dot(u,sqrt_singular_val)
    v = dot(sqrt_singular_val,v)
    weights_full = u.transpose()
    low_rank_a = weights_full[0:rank]
    low_rank_b = v[0:rank,:].transpose()
    return (low_rank_a, low_rank_b, rank)