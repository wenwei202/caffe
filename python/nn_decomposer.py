import caffe
import re
from pittnuts import *
import os
import matplotlib.pyplot as plt
import argparse
import caffeparser
import caffe_apps
import json
import numpy as np

def load_config(config_file):
    with open(config_file, 'r') as fp:
        conf = json.load(fp)
    return conf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prototxt', type=str, required=True,help="The original network structure.")
    parser.add_argument('--caffemodel', type=str, required=True,help="The trained caffemodel to be decomposed.")
    parser.add_argument('--rankratio', type=float, required=False,help="The ratio of reserved information based on eigenvalues.")
    parser.add_argument('--ranks', type=str, required=False,help="The reserved rank in each conv layers.")
    parser.add_argument('--rank_config', type=str, required=False, help="The json file to configure ranks in layers.")
    parser.add_argument('--except_layers', type=str, required=False, help="The Json file that excludes some conv layers from decomposition.")
    parser.add_argument('--lra_type', type=str, required=False, help="The type of Low rank approximation: pca (default), svd and kmeans")
    parser.add_argument('--enable_fc', dest='enable_fc', action='store_true')
    parser.add_argument('--no-enable_fc', dest='enable_fc', action='store_false')
    parser.set_defaults(enable_fc=False)
    args = parser.parse_args()
    prototxt = args.prototxt
    caffemodel = args.caffemodel

    rankratio = args.rankratio
    ranks = args.ranks
    rank_config = args.rank_config
    except_layers = args.except_layers

    rank_param_num = (None!=ranks) + (None!=rankratio) + (None!=rank_config)
    if 1!=rank_param_num:
        print "Please use one of --rankratio, --ranks or --rank_config"
        exit()
    if None != ranks:
        ranks = args.ranks.split(',')
        for i in range(0,len(ranks)):
            ranks[i] = int(ranks[i])
    if None != rank_config:
        rank_config = load_config(args.rank_config)
    if None != except_layers:
        except_layers = load_config(args.except_layers)

    lra_type = "pca"
    if None!=args.lra_type:
        lra_type = args.lra_type

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
    rank_idx = -1
    rank_info = ""
    for cur_layer in loop_layers:
        layer_idx += 1
        layer_name = cur_layer.name
        if 'Convolution' == cur_layer.type and ( (None==rank_config) or (layer_name in rank_config.keys()) ) and ((None==except_layers) or (layer_name not in except_layers) ):
            rank_idx += 1
            group = cur_layer.convolution_param.group
            assert (1==group) or (None==rankratio)
            weights = orig_net.params[layer_name][0].data
            filter_num = weights.shape[0]
            chan_num = weights.shape[1]
            kernel_h = weights.shape[2]
            kernel_w = weights.shape[3]
            # kernel_size = kernel_h * kernel_w
            # # decompose the weights
            # weights_pca = weights.reshape((filter_num, chan_num * kernel_size)).transpose()
            # weights_mean = mean(weights_pca,axis=0)
            # weights_pca, eig_vecs, eig_values = pca(weights_pca)
            # reconstruction_weight = copy.copy(weights_pca)
            # if None != rankratio:
            #     rank = caffe_apps.rank_by_ratio(eig_values, rankratio)
            # elif None != ranks:
            #     rank = ranks[rank_idx]
            # rank_info = rank_info + "{}\t{}/{} filters\n".format(layer_name, rank,filter_num)
            # shift_vals = dot(weights_mean,eig_vecs[:,rank:])
            # weights_full = weights_pca.transpose().reshape((filter_num, chan_num, kernel_h, kernel_w))
            # low_rank_filters = weights_full[0:rank]
            # linear_combinations = eig_vecs[:,0:rank].reshape((filter_num,rank,1,1))
            cur_rank = None
            if None != ranks:
                cur_rank = ranks[rank_idx]
                cur_rank = cur_rank / group * group
            elif None != rank_config:
                cur_rank = rank_config[layer_name]
                cur_rank = cur_rank / group * group
            final_rank = 0
            if None!=rankratio:
                rank = -1
                if "pca" == lra_type:
                    low_rank_filters, linear_combinations, rank = caffe_apps.filter_pca(filter_weights=weights,
                                                                                        ratio=rankratio,
                                                                                        rank= cur_rank)
                elif "svd" == lra_type:
                    low_rank_filters, linear_combinations, rank = caffe_apps.filter_svd(filter_weights=weights,
                                                                                        ratio=rankratio,
                                                                                        rank=cur_rank)
                else:
                    print "Unsupported --lra_type with --rankratio"
                    exit()
                final_rank = rank
            else:
                group_size1 = cur_rank/group
                group_size2 = filter_num/group
                low_rank_filters = np.zeros((cur_rank, chan_num, kernel_h, kernel_w))
                linear_combinations = np.zeros((filter_num, group_size1, 1, 1))
                for g in range(0,group):
                    rank = -1
                    if "pca" == lra_type:
                        low_rank_filters[g * group_size1:(g + 1) * group_size1], linear_combinations[g * group_size2:( g + 1) * group_size2], rank =  \
                            caffe_apps.filter_pca(filter_weights=weights[g * group_size2:(g + 1) * group_size2],
                                                  ratio=None,
                                                  rank=group_size1)
                    elif "svd" == lra_type:
                        low_rank_filters[g * group_size1:(g + 1) * group_size1], linear_combinations[g * group_size2:( g + 1) * group_size2], rank = \
                            caffe_apps.filter_svd(filter_weights=weights[g * group_size2:(g + 1) * group_size2],
                                                  ratio=None,
                                                  rank=group_size1)
                    elif "kmeans" == lra_type:
                        low_rank_filters[g * group_size1:(g + 1) * group_size1], linear_combinations[g * group_size2:( g + 1) * group_size2], rank = \
                            caffe_apps.filter_kmeans(filter_weights=weights[g * group_size2:(g + 1) * group_size2],
                                                  rank=group_size1)
                    final_rank += rank

            rank_info = rank_info + "{}\t{}/{} filters\n".format(layer_name, final_rank, filter_num)

            cur_layer_param = net_msg.layer._values.pop(layer_idx)
            # generate the low rank conv layer and move bias to the next layer
            low_rank_layer = caffe.proto.caffe_pb2.LayerParameter()
            low_rank_layer.CopyFrom(cur_layer_param)
            low_rank_layer.name = low_rank_layer.name+"_lowrank"
            low_rank_layer.top._values[0] = low_rank_layer.top._values[0] + "_lowrank"
            assert len(low_rank_layer.param._values)<=2
            if len(low_rank_layer.param._values)==2:
                low_rank_layer.param._values.pop()
            low_rank_layer.convolution_param.num_output = final_rank
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

            # reconstruction_weight[:,rank:] = tile(shift_vals,(reconstruction_weight.shape[0],1))
            # reconstruction_weight = subtract(weights_pca, reconstruction_weight)
            # reconstruction_error = dot(reconstruction_weight.reshape((1,-1)),reconstruction_weight.reshape((-1,1)))
            # rank_info = rank_info + "reconstruction_error={}\n".format(reconstruction_error)
            # rank_info = rank_info + "sumofeig={}\n".format(sum(eig_values[rank:])*(reconstruction_weight.shape[0]-1))

            # insert and add idx
            net_msg.layer._values.insert(layer_idx,low_rank_layer)
            layer_idx += 1
            net_msg.layer._values.insert(layer_idx, linear_layer)
        elif 'InnerProduct' == cur_layer.type and args.enable_fc and ( (None==rank_config) or (layer_name in rank_config.keys()) ) and ((None==except_layers) or (layer_name not in except_layers) ):
            rank_idx += 1
            weights = orig_net.params[layer_name][0].data
            cur_rank = None
            if None != ranks:
                cur_rank = ranks[rank_idx]
            elif None != rank_config:
                cur_rank = rank_config[layer_name]
            if "pca" == lra_type:
                low_rank_a, low_rank_b, final_rank = caffe_apps.fc_pca(fc_weights=weights,
                                                                 ratio=rankratio,
                                                                 rank=cur_rank)
            elif "svd" == lra_type:
                low_rank_a, low_rank_b, final_rank = caffe_apps.fc_svd(fc_weights=weights,
                                                                 ratio=rankratio,
                                                                 rank=cur_rank)
            else:
                print "Unsupported --lra_type with --rankratio"
                exit()

            rank_info = rank_info + "{}\t{}/{} neurons\n".format(layer_name, final_rank, weights.shape[0])

            cur_layer_param = net_msg.layer._values.pop(layer_idx)
            # generate the low rank conv layer and move bias to the next layer
            low_rank_layer = caffe.proto.caffe_pb2.LayerParameter()
            low_rank_layer.CopyFrom(cur_layer_param)
            low_rank_layer.name = low_rank_layer.name + "_lowrank"
            low_rank_layer.top._values[0] = low_rank_layer.top._values[0] + "_lowrank"
            assert len(low_rank_layer.param._values) <= 2
            if len(low_rank_layer.param._values) == 2:
                low_rank_layer.param._values.pop()
            low_rank_layer.inner_product_param.num_output = final_rank
            if low_rank_layer.inner_product_param.HasField("bias_filler"):
                low_rank_layer.inner_product_param.ClearField("bias_filler")
            bias_flag = low_rank_layer.inner_product_param.bias_term
            low_rank_layer.inner_product_param.bias_term = False
            new_parameters[low_rank_layer.name] = {0: low_rank_a[:]}

            linear_layer = caffe.proto.caffe_pb2.LayerParameter()
            linear_layer.CopyFrom(cur_layer_param)
            linear_layer.name = linear_layer.name + "_linear"
            linear_layer.bottom._values[0] = low_rank_layer.top._values[0]
            if bias_flag:
                new_parameters[linear_layer.name] = {0: low_rank_b[:],
                                                     1: orig_net.params[layer_name][1].data[:]}
            else:
                new_parameters[linear_layer.name] = {0: low_rank_b[:]}

            # insert and add idx
            net_msg.layer._values.insert(layer_idx, low_rank_layer)
            layer_idx += 1
            net_msg.layer._values.insert(layer_idx, linear_layer)
        else:
            if layer_name in orig_net.params:
                cur_param = {}
                for idx in range(0,len(orig_net.params[layer_name])):
                    cur_param[idx]=orig_net.params[layer_name][idx].data[:]
                new_parameters[layer_name] = cur_param

    # save the new network proto
    #file_split = os.path.splitext(prototxt)
    filepath_network = prototxt+".lowrank.prototxt" # DO NOT CHANGE THE SUFFIX - Other scripts depend on this
    file = open(filepath_network, "w")
    if not file:
        raise IOError("ERROR (" + filepath_network + ")!")
    file.write(str(net_msg))
    file.close()
    #print net_msg

    print rank_info

    # open and generate the caffemodel
    dst_net = caffe.Net(filepath_network, caffe.TRAIN)
    for key,val in new_parameters.iteritems():
        for keykey,valval in val.iteritems():
            dst_net.params[key][keykey].data[:] = valval[:]

    #file_split = os.path.splitext(caffemodel)
    filepath_caffemodel = caffemodel + '.lowrank.caffemodel.h5' # DO NOT CHANGE THE SUFFIX - Other scripts depend on this
    dst_net.save_hdf5(filepath_caffemodel)

    print "Saved as {}".format(filepath_network)
    print "Saved as {}".format(filepath_caffemodel)
    print "Done!"
