__author__ = 'pittnuts'
'''
add connectivity_mode to the network prototxt
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
	parser.add_argument('--connectivity_mode', type=str, required=True)
	args = parser.parse_args()
	prototxt_file = args.net
	connectivity_mode = args.connectivity_mode
	caffe.set_device(0)
	caffe.set_mode_gpu()
	
	net_parser = caffeparser.CaffeProtoParser(prototxt_file)
	net_msg = net_parser.readProtoNetFile()
	loop_layers = net_msg.layer[:] #adding : implicitly makes a copy to avoid being modified in the loop
	for cur_layer in loop_layers:
	    if 'Convolution'==cur_layer.type:
		if "DISCONNECTED_ELTWISE"==connectivity_mode:
		        cur_layer.connectivity_mode = caffe_pb2.LayerParameter.DISCONNECTED_ELTWISE
		elif "DISCONNECTED_GRPWISE"==connectivity_mode:
        		cur_layer.connectivity_mode = caffe_pb2.LayerParameter.DISCONNECTED_GRPWISE
		elif "CONNECTED"==connectivity_mode:
        		cur_layer.connectivity_mode = caffe_pb2.LayerParameter.CONNECTED
		else:
			print("Unexpected connectivity_mode {}".format(connectivity_mode))
			exit()
	
	file_split = os.path.splitext(prototxt_file)
	filepath = file_split[0]+'_'+connectivity_mode+file_split[1]
	file = open(filepath, "w")
	if not file:
	    raise IOError("ERROR (" + filepath + ")!")
	file.write(str(net_msg))
	file.close()
	print "Saved as {}".format(filepath)
