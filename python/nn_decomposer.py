

import caffe
import re
from pittnuts import *
import os
import matplotlib.pyplot as plt
import argparse
import caffeparser

def rank_by_ratio(eig_values,ratio):
    assert ratio<=1 and ratio>0
    eig_sum = sum(eig_values)
    for i in range(1, eig_values.size):
        eig_values[i] = eig_values[i] + eig_values[i - 1]
    eig_values = eig_values / eig_sum
    # return the rank that keeps ratio information
    for i in range(0, eig_values.size):
        if eig_values[i]>=ratio:
            return i+1
    return eig_values.size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prototxt', type=str, required=True)
    parser.add_argument('--caffemodel', type=str, required=True)
    parser.add_argument('--rankratio', type=float, required=False)
    parser.add_argument('--ranks', type=str, required=False)
    args = parser.parse_args()
    prototxt = args.prototxt
    caffemodel = args.caffemodel
    rankratio = args.rankratio
    ranks = args.ranks
    if None!=ranks and None!=rankratio:
        print "Please use either --rankratio or --ranks"
        exit()
    elif None==ranks and None==rankratio:
        print "Using default --rankratio 0.95"
        rankratio = 0.95
    elif None==rankratio and None != ranks:
        ranks = args.ranks.split(',')
        for i in range(0,len(ranks)):
            ranks[i] = int(ranks[i])

    net_parser = caffeparser.CaffeProtoParser(prototxt)
    net_msg = net_parser.readProtoNetFile()

    caffe.set_mode_cpu()
    # GPU mode
    #caffe.set_device(0)
    #caffe.set_mode_gpu()

    orig_net = caffe.Net(prototxt,caffemodel, caffe.TEST)

    print("blobs {}\nparams {}".format(orig_net.blobs.keys(), orig_net.params.keys()))

    loop_layers = net_msg.layer[:]  # adding : implicitly makes a copy to avoid being modified in the loop
    layer_idx = -1
    new_parameters = {}
    conv_idx = -1
    for cur_layer in loop_layers:
        layer_idx += 1
        layer_name = cur_layer.name
        if 'Convolution' == cur_layer.type:
            conv_idx += 1
            assert 1==cur_layer.convolution_param.group
            weights = orig_net.params[layer_name][0].data
            filter_num = weights.shape[0]
            chan_num = weights.shape[1]
            kernel_h = weights.shape[2]
            kernel_w = weights.shape[3]
            kernel_size = kernel_h * kernel_w
            # decompose the weights
            weights_pca = weights.reshape((filter_num, chan_num * kernel_size)).transpose()
            weights_pca, eig_vecs, eig_values = pca(weights_pca)
            if None != rankratio:
                rank = rank_by_ratio(eig_values, rankratio)
            elif None != ranks:
                rank = ranks[conv_idx]
            print "{}\t{}/{} filters".format(layer_name, rank,filter_num)
            weights_pca = weights_pca.transpose().reshape(filter_num, chan_num, kernel_h, kernel_w)
            low_rank_filters = weights_pca[0:rank]
            linear_combinations = eig_vecs[:,0:rank].reshape(filter_num,rank,1,1)

            cur_layer_param = net_msg.layer._values.pop(layer_idx)
            # generate the low rank conv layer and remove bias
            low_rank_layer = caffe.proto.caffe_pb2.LayerParameter()
            low_rank_layer.CopyFrom(cur_layer_param)
            low_rank_layer.name = low_rank_layer.name+"_lowrank"
            low_rank_layer.top._values[0] = low_rank_layer.top._values[0] + "_lowrank"
            assert len(low_rank_layer.param._values)<=2
            if len(low_rank_layer.param._values)==2:
                low_rank_layer.param._values.pop()
            low_rank_layer.convolution_param.num_output = rank
            if low_rank_layer.convolution_param.HasField("bias_filler"):
                low_rank_layer.convolution_param.ClearField("bias_filler")
            bias_flag = low_rank_layer.convolution_param.bias_term
            low_rank_layer.convolution_param.bias_term = False
            new_parameters[low_rank_layer.name] = {0:low_rank_filters[:]}

            linear_layer = caffe.proto.caffe_pb2.LayerParameter()
            linear_layer.CopyFrom(cur_layer_param)
            linear_layer.name = linear_layer.name + "_linear"
            linear_layer.bottom._values[0] = low_rank_layer.top._values[0]
            assert len(linear_layer.convolution_param.kernel_size._values) == 1
            linear_layer.convolution_param.kernel_size._values[0] = 1
            if len(linear_layer.convolution_param.stride._values)>0:
                assert len(linear_layer.convolution_param.stride._values) == 1
                linear_layer.convolution_param.stride._values[0] = 1
            if len(linear_layer.convolution_param.pad._values)>0:
                assert len(linear_layer.convolution_param.pad._values) == 1
                linear_layer.convolution_param.pad._values[0] = 0
            if len(linear_layer.convolution_param.dilation._values)>0:
                assert len(linear_layer.convolution_param.dilation._values) == 1
                linear_layer.convolution_param.dilation._values[0] = 1
            if bias_flag:
                new_parameters[linear_layer.name] = {0: linear_combinations[:],
                                                     1: orig_net.params[layer_name][1].data[:]}
            else:
                new_parameters[linear_layer.name] = {0: linear_combinations[:]}


            # insert and add idx
            net_msg.layer._values.insert(layer_idx,low_rank_layer)
            layer_idx += 1
            net_msg.layer._values.insert(layer_idx, linear_layer)
        else:
            if layer_name in orig_net.params:
                cur_param = {}
                for idx in range(0,len(orig_net.params[layer_name])):
                    cur_param[idx]=orig_net.params[layer_name][idx].data[:]
                new_parameters[layer_name] = cur_param

    # save the new network proto
    file_split = os.path.splitext(prototxt)
    filepath_network = file_split[0] + '_lowrank' + file_split[1]
    file = open(filepath_network, "w")
    if not file:
        raise IOError("ERROR (" + filepath_network + ")!")
    file.write(str(net_msg))
    file.close()
    #print net_msg

    # open and generate the caffemodel
    dst_net = caffe.Net(filepath_network, caffe.TRAIN)
    for key,val in new_parameters.iteritems():
        for keykey,valval in val.iteritems():
            dst_net.params[key][keykey].data[:] = valval[:]

    file_split = os.path.splitext(caffemodel)
    filepath_caffemodel = file_split[0] + '.lowrank.caffemodel'
    dst_net.save(filepath_caffemodel)

    print "Saved as {}".format(filepath_network)
    print "Saved as {}".format(filepath_caffemodel)
    print "Done!"
