__author__ = 'pittnuts'
import caffe
import re
from pittnuts import *
from caffe_apps import *
import os
import matplotlib.pyplot as plt
import matplotlib
import argparse
import caffeparser
import numpy as np
import sklearn.preprocessing as skp
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
g_colors = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

# --prototxt models/bvlc_reference_caffenet/deploy.prototxt --origimodel models/bvlc_reference_caffenet/caffenet_0.57368.caffemodel --tunedmodel models/bvlc_reference_caffenet/
# --prototxt examples/mnist/lenet.prototxt --origimodel examples/mnist/lenet_0.9917.caffemodel --tunedmodel examples/mnist/lenet_iter_10000.caffemodel
# --prototxt examples/mnist/lenet_.prototxt --origimodel examples/mnist/lenet_0.9917.caffemodel --tunedmodel examples/mnist/lenet_iter_10000.caffemodel
# --prototxt examples/cifar10/cifar10_full.prototxt --origimodel examples/cifar10/cifar10_full_iter_240000_0.8201.caffemodel.h5 --tunedmodel examples/cifar10/cifar10_full_iter_240000.caffemodel.h5
def print_eig_info(eig_values,style,percent=0.95):
    eig_sum = sum(eig_values)
    #print eig_values
    for i in range(1, eig_values.size):
        eig_values[i] = eig_values[i] + eig_values[i - 1]
    eig_values = eig_values / eig_sum
    if ''==style:
        style='-ro'
    plt.plot(eig_values,style)
    for i in range(0, eig_values.size):
        if eig_values[i]>percent:
            print "{} / {} is more than {} of eigenvalue sum".format(i+1,eig_values.size,percent)
            break

def show_filters(net,layername ,filt_min ,filt_max):
    weights = net.params[layername][0].data
    if len(weights.shape) < 3:
        return
    chan_num = weights.shape[1]
    filter_num = weights.shape[0]
    #display_region_size = ceil(sqrt(filter_num))
    rgb = (chan_num==3)
    plt.figure()
    if rgb:
        for n in range(filter_num/2):
            plt.subplot(6, 16,  n + 1)
            img = (weights[n, :].transpose((1,2,0)) - filt_min)/(filt_max-filt_min)
            plt.imshow(img,  interpolation='none')
            plt.tick_params(which='both', labelbottom='off', labelleft='off', bottom='off', top='off', left='off', right='off')
            ax = plt.gca()
            if n<11:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
    else:
        #c_ordered=(3,10,12,0,1,2,4,5,6,7,8,9,11,13,14,15,16,17,18,19)
        for c in range(min(20,chan_num)):
            #if sum(abs(weights[:,c,:,:]))>0:
                #print "{}-th channel is usefull".format(c)
                for n in range(filter_num):
                    #plt.subplot((int)(display_region_size),(int)(display_region_size),n+1)
                    plt.subplot(chan_num,filter_num, filter_num*c + n + 1)
                    #if sum(abs(weights[n,c]))>0:
                    plt.imshow(weights[n,c],vmin=filt_min,vmax=filt_max,cmap=plt.get_cmap('Greys'),interpolation='none')
                    #plt.imshow(weights[n,c_ordered[c]], vmin=filt_min, vmax=filt_max, cmap=plt.get_cmap('Greys'), interpolation='none')
                    plt.tick_params(which='both',labelbottom='off',labelleft='off',bottom='off',top='off',left='off',right='off')

def show_2Dfilter_pca(net,layername,showit=False):
    weights = net.params[layername][0].data
    if len(weights.shape) < 3:
        return
    filter_num = weights.shape[0]
    chan_num = weights.shape[1]
    kernel_h = weights.shape[2]
    kernel_w = weights.shape[3]
    kernel_size = kernel_h*kernel_w

    weights_pca = weights.reshape((chan_num*filter_num, kernel_size)).transpose()
    weights_pca, eig_vecs, eig_values = pca(weights_pca)
    #print_eig_info(eig_values)
    if showit:
        weights_pca = weights_pca.transpose().reshape(filter_num,chan_num,kernel_h,kernel_w)
        filt_max = abs(weights_pca).max()
        filt_min = -filt_max
        #eig_vecs = eig_vecs.transpose().reshape(kernel_size,kernel_h,kernel_w)
        plt.figure()
        for c in range(min(20, chan_num)):
            for n in range(filter_num):
                plt.subplot(chan_num, filter_num, filter_num * c + n + 1)
                plt.imshow(weights_pca[n, c], vmin=filt_min, vmax=filt_max, cmap=plt.get_cmap('Greys'), interpolation='none')
                plt.tick_params(which='both', labelbottom='off', labelleft='off', bottom='off', top='off', left='off',right='off')

def show_filter_channel_pca(net,layername,style,display_channel=False):
    weights = net.params[layername][0].data
    if len(weights.shape) < 3:
        return
    filter_num = weights.shape[0]
    chan_num = weights.shape[1]
    kernel_h = weights.shape[2]
    kernel_w = weights.shape[3]
    kernel_size = kernel_h*kernel_w

    # filter-wise
    print layername+" analyzing filter-wise:"
    weights_pca = weights.reshape((filter_num, chan_num*kernel_size)).transpose()
    weights_pca, eig_vecs, eig_values = pca(weights_pca)
    print_eig_info(eig_values,style)
    weights_pca = weights_pca.transpose().reshape(filter_num,chan_num,kernel_h,kernel_w)
    if display_channel:
        # channel-wise
        print layername+" analyzing channel-wise:"
        weights_pca = weights_pca.transpose((1,0,2,3)).reshape((chan_num,  filter_num* kernel_size)).transpose()
        weights_pca, eig_vecs, eig_values = pca(weights_pca)
        print_eig_info(eig_values,style)

def show_filter_pca_projection(net,layername):
    weights = net.params[layername][0].data
    if len(weights.shape) < 3:
        return
    filter_num = weights.shape[0]
    chan_num = weights.shape[1]
    kernel_h = weights.shape[2]
    kernel_w = weights.shape[3]
    kernel_size = kernel_h*kernel_w

    weights_norm = weights.reshape((filter_num, chan_num * kernel_size))
    weights_norm = skp.normalize(weights_norm)

    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters).fit(weights_norm)


    weights_pca, eig_vecs, eig_values = pca(weights_norm)
    colors = np.random.rand(n_clusters)
    plt.scatter(weights_pca[:,0], weights_pca[:,1], c=colors[kmeans.labels_], s=200,alpha=0.5)
    plt.show()


def show_filter_lda_projection(net,layername):
    weights = net.params[layername][0].data
    if len(weights.shape) < 3:
        return
    filter_num = weights.shape[0]
    chan_num = weights.shape[1]
    kernel_h = weights.shape[2]
    kernel_w = weights.shape[3]
    kernel_size = kernel_h*kernel_w

    weights_norm = weights.reshape((filter_num, chan_num * kernel_size))
    weights_norm = skp.normalize(weights_norm)

    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters).fit(weights_norm)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(weights_norm, kmeans.labels_).transform(weights_norm)

    colors = np.random.rand(n_clusters)
    plt.scatter(X_r2[:, 0], X_r2[:, 1], c=g_colors[kmeans.labels_], s=300, alpha=0.5)

    plt.show()


def get_angle_sum(net,layername):
    weights = net.params[layername][0].data
    if len(weights.shape) < 3:
        return
    filter_num = weights.shape[0]
    chan_num = weights.shape[1]
    kernel_h = weights.shape[2]
    kernel_w = weights.shape[3]
    kernel_size = kernel_h*kernel_w

    weights_norm = weights.reshape((filter_num, chan_num * kernel_size))
    weights_norm = skp.normalize(weights_norm)
    angle_sum = 0
    for idx1 in range(0,filter_num-1):
        for idx2 in range(idx1+1, filter_num):
            angle_sum += angle(weights_norm[idx1],weights_norm[idx2])

    return angle_sum

def show_filter_shapes(net, layername):
    weights = net.params[layername][0].data
    if len(weights.shape) < 3:
        return
    chan_num = weights.shape[1]
    filter_num = weights.shape[0]
    weights = abs(weights)
    weights = sum(weights,axis=0)!=0
    plt.figure()
    for c in range(min(20, chan_num)):
        plt.subplot(chan_num, 1, c + 1)
        plt.imshow(weights[c], vmin=0, vmax=1, cmap=plt.get_cmap('Greys'), interpolation='none')
        plt.tick_params(which='both', labelbottom='off', labelleft='off', bottom='off', top='off', left='off', right='off')

# get the maximum abs weight
def get_max_weight(orig_net,tuned_net,layer_name):
    weight_scope = 0
    weights_orig = orig_net.params[layer_name][0].data
    weights_tuned = tuned_net.params[layer_name][0].data
    max_val = abs(weights_orig).max()
    if max_val > weight_scope:
        weight_scope = max_val
    max_val = abs(weights_tuned).max()
    if max_val > weight_scope:
        weight_scope = max_val
    return weight_scope


if __name__ == "__main__":

    matplotlib.rcParams.update({'font.size': 22})
    parser = argparse.ArgumentParser()
    parser.add_argument('--prototxt', type=str, required=True)
    parser.add_argument('--origimodel', type=str, required=True)
    parser.add_argument('--tunedmodel', type=str, required=True)
    args = parser.parse_args()
    prototxt = args.prototxt #"models/eilab_reference_sparsenet/train_val_scnn.prototxt"
    original_caffemodel = args.origimodel # "models/eilab_reference_sparsenet/eilab_reference_sparsenet.caffemodel"
    fine_tuned_caffemodel = args.tunedmodel # "/home/wew57/2bincaffe/models/eilab_reference_sparsenet/sparsenet_train_iter_30000.caffemodel"
    net_parser = caffeparser.CaffeProtoParser(prototxt)
    net_msg = net_parser.readProtoNetFile()

    caffe.set_mode_cpu()
    # GPU mode
    #caffe.set_device(1)
    #caffe.set_mode_gpu()

    orig_net = caffe.Net(prototxt,original_caffemodel, caffe.TEST)
    tuned_net = caffe.Net(prototxt,fine_tuned_caffemodel, caffe.TEST)

    print("blobs {}\nparams {}".format(orig_net.blobs.keys(), orig_net.params.keys()))
    print("blobs {}\nparams {}".format(tuned_net.blobs.keys(), tuned_net.params.keys()))

    #show_2Dfilter_pca(orig_net, 'conv1',True)
    #show_2Dfilter_pca(tuned_net, 'conv1',True)
    #show_2Dfilter_pca(orig_net, 'conv2')
    #show_2Dfilter_pca(tuned_net, 'conv2')
    #show_2Dfilter_pca(orig_net, 'conv1')
    #show_2Dfilter_pca(orig_net, 'conv2')
    #show_2Dfilter_pca(orig_net, 'conv3')
    #show_2Dfilter_pca(orig_net, 'conv4')
    #show_2Dfilter_pca(orig_net, 'conv5')

    #show_filter_channel_pca(orig_net, 'conv2','--r')
    #show_filter_channel_pca(tuned_net, 'conv2','-r')
    #show_filter_channel_pca(orig_net, 'conv2','--g')
    #show_filter_channel_pca(tuned_net, 'conv2','-g')
    #show_filter_channel_pca(orig_net, 'conv3', '--b')
    #show_filter_channel_pca(tuned_net, 'conv3', '-b')
    #show_filter_channel_pca(orig_net, 'conv3')
    #show_filter_channel_pca(tuned_net, 'conv3')

    weights_tmp = orig_net.params['conv1'][0].data[:]
    weights_tmp[:],tmp1,tmp2 = filter_pca(weights_tmp, rank=weights_tmp.shape[0])
    weight_scope = abs(weights_tmp).max()
    show_filters(orig_net,'conv1',-weight_scope, weight_scope)
    weights_tmp = tuned_net.params['conv1'][0].data[:]
    weights_tmp[:],tmp1,tmp2 = filter_pca(weights_tmp, rank=weights_tmp.shape[0])
    weight_scope = abs(weights_tmp).max()
    show_filters(tuned_net, 'conv1', -weight_scope, weight_scope)
    #plt.figure()
    #print get_angle_sum(orig_net, 'conv1')
    #print get_angle_sum(tuned_net, 'conv1')
    #print get_angle_sum(orig_net, 'conv2')
    #print get_angle_sum(tuned_net, 'conv2')
    #print get_angle_sum(orig_net, 'conv3')
    #print get_angle_sum(tuned_net, 'conv3')

    #show_filter_lda_projection(orig_net, 'conv3')
    #show_filter_lda_projection(tuned_net, 'conv3')

    plt.show()