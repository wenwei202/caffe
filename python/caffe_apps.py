__author__ = 'caffe'

import numpy as np
import matplotlib.pyplot as plt
from pittnuts import *
import copy

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

def filter_pca(filter_weights,ratio=None,rank=None):
    filter_num = filter_weights.shape[0]
    chan_num = filter_weights.shape[1]
    kernel_h = filter_weights.shape[2]
    kernel_w = filter_weights.shape[3]
    kernel_size = kernel_h * kernel_w
    # decompose the weights
    weights_pca = filter_weights.reshape((filter_num, chan_num * kernel_size)).transpose()
    weights_pca, eig_vecs, eig_values = pca(weights_pca)
    if None != ratio:
        rank = rank_by_ratio(eig_values, ratio)
    weights_full = weights_pca.transpose().reshape((filter_num, chan_num, kernel_h, kernel_w))
    low_rank_filters = weights_full[0:rank]
    linear_combinations = eig_vecs[:, 0:rank].reshape((filter_num, rank, 1, 1))
    return (low_rank_filters,linear_combinations,rank)