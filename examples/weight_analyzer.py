__author__ = 'pittnuts'
import caffe
import re
from pittnuts import *
import os
import matplotlib.pyplot as plt
import argparse
import caffeparser
# --prototxt models/bvlc_reference_caffenet/deploy.prototxt --origimodel models/bvlc_reference_caffenet/caffenet_0.57368.caffemodel --tunedmodel models/bvlc_reference_caffenet/
# --prototxt examples/mnist/lenet.prototxt --origimodel examples/mnist/lenet_0.9912.caffemodel --tunedmodel examples/mnist/
# --prototxt examples/cifar10/cifar10_full.prototxt --origimodel examples/cifar10/cifar10_full_iter_300000_0.8212.caffemodel --tunedmodel examples/cifar10/cifar10_full_grouplasso_iter_60000.caffemodel
def print_eig_info(eig_values,percent=0.95):
    eig_sum = sum(eig_values)
    #print eig_values
    for i in range(1, eig_values.size):
        eig_values[i] = eig_values[i] + eig_values[i - 1]
    eig_values = eig_values / eig_sum
    for i in range(1, eig_values.size):
        if eig_values[i]>percent:
            print "{} / {} is more than {} of eigenvalue sum".format(i+1,eig_values.size,percent)
            break

def show_filters(net,layername ,filt_min ,filt_max):
    rgb = False
    weights = net.params[layername][0].data
    if len(weights.shape) < 3:
        return
    chan_num = weights.shape[1]
    filter_num = weights.shape[0]
    #display_region_size = ceil(sqrt(filter_num))
    rgb = (chan_num==3)
    plt.figure()
    if rgb:
        for n in range(filter_num):
            plt.subplot(6, 16,  n + 1)
            img = (weights[n, :].transpose((1,2,0)) - filt_min)/(filt_max-filt_min)
            plt.imshow(img,  interpolation='none')
            plt.tick_params(which='both', labelbottom='off', labelleft='off', bottom='off', top='off', left='off', right='off')
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
    print_eig_info(eig_values)
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

def show_filter_channel_pca(net,layername):
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
    print_eig_info(eig_values)
    weights_pca = weights_pca.transpose().reshape(filter_num,chan_num,kernel_h,kernel_w)
    # channel-wise
    print layername+" analyzing channel-wise:"
    weights_pca = weights_pca.transpose((1,0,2,3)).reshape((chan_num,  filter_num* kernel_size)).transpose()
    weights_pca, eig_vecs, eig_values = pca(weights_pca)
    print_eig_info(eig_values)

    #weights_pca = weights_pca.transpose().reshape(chan_num, filter_num, kernel_h, kernel_w).transpose((1,0,2,3))
    #filt_max = abs(weights_pca).max()
    #filt_min = -filt_max
    ##eig_vecs = eig_vecs.transpose().reshape(kernel_size,kernel_h,kernel_w)
    #plt.figure()
    #for c in range(min(20, chan_num)):
    #    for n in range(filter_num):
    #        plt.subplot(chan_num, filter_num, filter_num * c + n + 1)
    #        plt.imshow(weights_pca[n, c], vmin=filt_min, vmax=filt_max, cmap=plt.get_cmap('Greys'), interpolation='none')
    #        plt.tick_params(which='both', labelbottom='off', labelleft='off', bottom='off', top='off', left='off',right='off')

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

    #show_2Dfilter_pca(orig_net, 'conv1')
    #show_2Dfilter_pca(tuned_net, 'conv1')
    #show_2Dfilter_pca(orig_net, 'conv2')
    #show_2Dfilter_pca(tuned_net, 'conv2')
    #show_2Dfilter_pca(orig_net, 'conv1')
    #show_2Dfilter_pca(orig_net, 'conv2')
    #show_2Dfilter_pca(orig_net, 'conv3')
    #show_2Dfilter_pca(orig_net, 'conv4')
    #show_2Dfilter_pca(orig_net, 'conv5')

    show_filter_channel_pca(orig_net, 'conv1')
    show_filter_channel_pca(tuned_net, 'conv1')
    show_filter_channel_pca(orig_net, 'conv2')
    show_filter_channel_pca(tuned_net, 'conv2')
    show_filter_channel_pca(orig_net, 'conv3')
    show_filter_channel_pca(tuned_net, 'conv3')

    #weight_scope = get_max_weight(orig_net, tuned_net, 'conv1')
    #show_filters(orig_net,'conv1',-weight_scope, weight_scope)
    #show_filters(tuned_net, 'conv1', -weight_scope, weight_scope)
    #show_filter_shapes(orig_net, 'conv1')
    #show_filter_shapes(tuned_net, 'conv1')


    #weight_scope = get_max_weight(orig_net, tuned_net, 'conv2')
    #show_filters(orig_net, 'conv2', -weight_scope, weight_scope)
    #show_filters(tuned_net, 'conv2', -weight_scope, weight_scope)
    #show_filter_shapes(orig_net, 'conv2')
    #show_filter_shapes(tuned_net, 'conv2')

    plt.show()