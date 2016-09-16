import numpy as np
import matplotlib.pyplot as plt
from scipy.io import *
from PIL import Image
import caffe
import sys
import lmdb
from caffe.proto import caffe_pb2
from pittnuts import *
import os
from caffe_apps import *
import caffeparser
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_template', type=str, required=True)
    parser.add_argument('--layer_type', type=str, required=True)
    parser.add_argument('--param_value', type=str, required=True)
    args = parser.parse_args()
    net_template = args.net_template
    layer_type = args.layer_type
    param_value = args.param_value

    caffe.set_mode_cpu()
    net_parser = caffeparser.CaffeProtoParser(net_template)
    net_msg = net_parser.readProtoNetFile()

    for cur_layer in net_msg.layer:
        if 'Sparsify' == cur_layer.type:
            cur_layer.sparsify_param.coef = float(param_value)

    # save
    dirname = os.path.dirname(net_template)
    filepath = dirname + "/generated.prototxt"
    file = open(filepath, "w")
    if not file:
        raise IOError("ERROR (" + filepath + ")!")
    file.write(str(net_msg))
    file.close()
