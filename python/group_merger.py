import caffe
import re
from pittnuts import *
import os
import matplotlib.pyplot as plt
import argparse
import caffeparser
import caffe_apps
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_alexnet', type=str, required=True,help="The original alexnet with group.")
    parser.add_argument('--split_alexnet', type=str, required=True, help="The split alexnet without group.")
    parser.add_argument('--caffemodel', type=str, required=True,help="The caffemodel of split alexnet.")
    args = parser.parse_args()
    original_alexnet = args.original_alexnet
    caffemodel = args.caffemodel
    split_alexnet = args.split_alexnet

    net_parser = caffeparser.CaffeProtoParser(original_alexnet)
    orig_net_msg = net_parser.readProtoNetFile()
    net_parser = caffeparser.CaffeProtoParser(split_alexnet)
    split_net_msg = net_parser.readProtoNetFile()

    caffe.set_mode_cpu()
    # GPU mode
    #caffe.set_device(0)
    #caffe.set_mode_gpu()

    src_net = caffe.Net(split_alexnet,caffemodel, caffe.TEST)

    print("blobs {}\nparams {}".format(src_net.blobs.keys(), src_net.params.keys()))

    loop_layers = orig_net_msg.layer[:]  # adding : implicitly makes a copy to avoid being modified in the loop
    layer_idx = -1
    new_parameters = {}
    for cur_layer in loop_layers:
        layer_idx += 1
        layer_name = cur_layer.name
        if 'Convolution' == cur_layer.type:
            if cur_layer.convolution_param.bias_term:
                new_parameters[layer_name] = {0: np.concatenate(( src_net.params[layer_name+"_group0"][0].data,  src_net.params[layer_name+"_group1"][0].data)),
                                              1: np.concatenate(( src_net.params[layer_name+"_group0"][1].data,  src_net.params[layer_name+"_group1"][1].data)) }
            else:
                new_parameters[layer_name] = {0: np.concatenate(
                    (src_net.params[layer_name + "_group0"][0].data, src_net.params[layer_name + "_group1"][0].data))}
        else:
            if layer_name in src_net.params:
                cur_param = {}
                for idx in range(0,len(src_net.params[layer_name])):
                    cur_param[idx]=src_net.params[layer_name][idx].data[:]
                new_parameters[layer_name] = cur_param

    # open and generate the caffemodel
    dst_net = caffe.Net(original_alexnet, caffe.TRAIN)
    for key,val in new_parameters.iteritems():
        for keykey,valval in val.iteritems():
            dst_net.params[key][keykey].data[:] = valval[:]

    #file_split = os.path.splitext(caffemodel)
    filepath_caffemodel = caffemodel + '.merge.caffemodel.h5'
    dst_net.save_hdf5(filepath_caffemodel)

    print "Saved as {}".format(filepath_caffemodel)
    print "Done!"
