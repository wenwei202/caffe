__author__ = 'pittnuts'
'''
add column-wise, row-wise and 2D-filter-wise sparsity to the network prototxt
'''
import sys
sys.path.append('/home/Pitt/github/caffe/python')
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import re
from numpy import *
import os
import caffeparser
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--net', type=str, required=True)
	args = parser.parse_args()
	prototxt_file = args.net
	caffe.set_device(0)
	caffe.set_mode_gpu()
	
	net_parser = caffeparser.CaffeProtoParser(prototxt_file)
	net_msg = net_parser.readProtoNetFile()
	loop_layers = net_msg.layer[:] #adding : implicitly makes a copy to avoid being modified in the loop
	for cur_layer in loop_layers:
	    if 'Convolution'==cur_layer.type:
		if len(cur_layer.param)==0:
			lr_param = caffe_pb2.ParamSpec()
			lr_param.lr_mult = 1.0
			lr_param.decay_mult = 1.0
			cur_layer.param._values.append(lr_param)
		kernel_size = cur_layer.convolution_param.kernel_size._values[0]
		blk_param = caffe_pb2.BlockGroupLassoSpec()
	        blk_param.xdimen = kernel_size*kernel_size
	        blk_param.ydimen = 1
		blk_param.block_decay_mult = 1.0
	        cur_layer.param._values[0].block_group_lasso._values.append(blk_param)	
		cur_layer.param._values[0].breadth_decay_mult = 1.0
		cur_layer.param._values[0].kernel_shape_decay_mult = 1.0
	
	file_split = os.path.splitext(prototxt_file)
	filepath = file_split[0]+'_ssl'+file_split[1]
	file = open(filepath, "w")
	if not file:
	    raise IOError("ERROR (" + filepath + ")!")
	file.write(str(net_msg))
	file.close()
	print "Saved as {}".format(filepath)
