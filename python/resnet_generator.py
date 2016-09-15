__author__ = 'wei wen'

import argparse
import caffeparser
import caffe
from caffe.proto import caffe_pb2
from numpy import *
import re
import pittnuts
import os

def add_conv_layer(net_msg,name,bottom,num_output,pad,kernel_size,stride,bias_term=True,learn_depth=False,input_channel=1,connectivity_mode=0):
    conv_layer = net_msg.layer.add()
    conv_layer.name = name
    conv_layer.type = 'Convolution'
    conv_layer.bottom._values.append(bottom)
    conv_layer.top._values.append(conv_layer.name)
    if 1==connectivity_mode:
        conv_layer.connectivity_mode = caffe_pb2.LayerParameter.DISCONNECTED_ELTWISE
    elif 2==connectivity_mode:
        conv_layer.connectivity_mode = caffe_pb2.LayerParameter.DISCONNECTED_GRPWISE
    # param info for weight and bias
    lr_param = caffe_pb2.ParamSpec()
    lr_param.lr_mult = 1
    if learn_depth:
        blk_param = caffe_pb2.BlockGroupLassoSpec()
        blk_param.xdimen = kernel_size*kernel_size*input_channel
        blk_param.ydimen = num_output
        lr_param.block_group_lasso._values.append(blk_param)
    conv_layer.param._values.append(lr_param)
    if bias_term:
        lr_param = caffe_pb2.ParamSpec()
        lr_param.lr_mult = 2
        conv_layer.param._values.append(lr_param)
    # conv parameters
    conv_layer.convolution_param.num_output = num_output
    conv_layer.convolution_param.pad._values.append(pad)
    conv_layer.convolution_param.kernel_size._values.append(kernel_size)
    conv_layer.convolution_param.stride._values.append(stride)
    conv_layer.convolution_param.weight_filler.type = 'msra'
    conv_layer.convolution_param.bias_term = bias_term
    if bias_term:
        conv_layer.convolution_param.bias_filler.type = 'constant'

def add_relu_layer(net_msg,name,bottom):
    relulayer = net_msg.layer.add()
    relulayer.name = name
    relulayer.type = 'ReLU'
    relulayer.bottom._values.append(bottom)
    relulayer.top._values.append(name)

def add_eltwise_add_layer(net_msg,name,bottom1,bottom2):
    eltlayer = net_msg.layer.add()
    eltlayer.name = name
    eltlayer.type = 'Eltwise'
    eltlayer.bottom._values.append(bottom1)
    eltlayer.bottom._values.append(bottom2)
    eltlayer.top._values.append(name)

def add_BN_layer(net_msg,name,bottom):
    # norm layer
    batchnormlayer = net_msg.layer.add()
    batchnormlayer.name = name+'_norm'
    batchnormlayer.type = 'BatchNorm'
    batchnormlayer.bottom._values.append(bottom)
    batchnormlayer.top._values.append(batchnormlayer.name)
    for i in range(0,3):
        lr_param = caffe_pb2.ParamSpec()
        lr_param.lr_mult = 0
        batchnormlayer.param._values.append(lr_param)
    # scale layer
    scalelayer = net_msg.layer.add()
    scalelayer.name = name+'_scale'
    scalelayer.type = 'Scale'
    scalelayer.bottom._values.append(batchnormlayer.name)
    scalelayer.top._values.append(name)

    lr_param = caffe_pb2.ParamSpec()
    lr_param.lr_mult = 1
    scalelayer.param._values.append(lr_param)
    lr_param = caffe_pb2.ParamSpec()
    lr_param.lr_mult = 2
    lr_param.decay_mult = 0
    scalelayer.param._values.append(lr_param)

    scalelayer.scale_param.bias_term = True
    scalelayer.scale_param.filler.type = 'msra'

def add_global_avg_pooling_layer(net_msg,name,bottom):
    glb_avg_pl_layer = net_msg.layer.add()
    glb_avg_pl_layer.name = name
    glb_avg_pl_layer.type = 'Pooling'
    glb_avg_pl_layer.bottom._values.append(bottom)
    glb_avg_pl_layer.top._values.append(name)
    glb_avg_pl_layer.pooling_param.pool = caffe_pb2.PoolingParameter.AVE
    glb_avg_pl_layer.pooling_param.global_pooling = True

def add_downsampling_layer(net_msg,name,bottom,stride):
    downsampling_layer = net_msg.layer.add()
    downsampling_layer.name = name
    downsampling_layer.type = 'Pooling'
    downsampling_layer.bottom._values.append(bottom)
    downsampling_layer.top._values.append(name)
    downsampling_layer.pooling_param.pool = caffe_pb2.PoolingParameter.MAX
    downsampling_layer.pooling_param.kernel_size = 1
    downsampling_layer.pooling_param.stride = stride

def add_ip_layer(net_msg,name,bottom,num):
    ip_layer = net_msg.layer.add()
    ip_layer.name = name
    ip_layer.type = 'InnerProduct'
    ip_layer.bottom._values.append(bottom)
    ip_layer.top._values.append(name)
    # param info for weight and bias
    lr_param = caffe_pb2.ParamSpec()
    lr_param.lr_mult = 1
    lr_param.decay_mult = 1
    ip_layer.param._values.append(lr_param)
    lr_param = caffe_pb2.ParamSpec()
    lr_param.lr_mult = 2
    lr_param.decay_mult = 0
    ip_layer.param._values.append(lr_param)
    # inner product parameters
    ip_layer.inner_product_param.num_output = num
    ip_layer.inner_product_param.weight_filler.type = 'msra'
    ip_layer.inner_product_param.bias_filler.type = 'constant'
    ip_layer.inner_product_param.bias_filler.value = 0.0

def add_accuracy_layer(net_msg,bottom):
    accuracy_layer = net_msg.layer.add()
    accuracy_layer.name = 'accuracy'
    accuracy_layer.type = 'Accuracy'
    accuracy_layer.bottom._values.append(bottom)
    accuracy_layer.bottom._values.append('label')
    accuracy_layer.top._values.append('accuracy')
    include_param = caffe_pb2.NetStateRule()
    include_param.phase = caffe_pb2.TEST
    accuracy_layer.include._values.append(include_param)

def add_loss_layer(net_msg,bottom):
    loss_layer = net_msg.layer.add()
    loss_layer.name = 'loss'
    loss_layer.type = 'SoftmaxWithLoss'
    loss_layer.bottom._values.append(bottom)
    loss_layer.bottom._values.append('label')
    loss_layer.top._values.append('loss')

def add_1st_res_layers(net_msg,name,bottom,learn_depth=False,connectivity_mode=0):
    # first layer
    add_conv_layer(net_msg,name=name+'_conv1',bottom=bottom,num_output=16,pad=1,kernel_size=3,stride=1,bias_term=False,learn_depth=learn_depth,input_channel=16,connectivity_mode=connectivity_mode)
    add_BN_layer(net_msg,name=name+'_bn1',bottom=name+'_conv1')
    add_relu_layer(net_msg,name=name+'_relu1',bottom=name+'_bn1')
    #second conv
    add_conv_layer(net_msg,name=name+'_conv2',bottom=name+'_relu1',num_output=16,pad=1,kernel_size=3,stride=1,bias_term=False,learn_depth=learn_depth,input_channel=16,connectivity_mode=connectivity_mode)
    add_BN_layer(net_msg,name=name+'_bn2',bottom=name+'_conv2')
    #add layer
    add_eltwise_add_layer(net_msg,name+'_add',bottom,name+'_bn2')
    #final relu
    add_relu_layer(net_msg,name=name,bottom=name+'_add')

def add_2nd_res_layers(net_msg,name,bottom,downsample=False,learn_depth=False,connectivity_mode=0):
    # first layer
    if downsample:
        add_conv_layer(net_msg,name=name+'_conv1',bottom=bottom,num_output=32,pad=1,kernel_size=3,stride=2,bias_term=False,learn_depth=learn_depth,input_channel=16,connectivity_mode=connectivity_mode)
    else:
        add_conv_layer(net_msg,name=name+'_conv1',bottom=bottom,num_output=32,pad=1,kernel_size=3,stride=1,bias_term=False,learn_depth=learn_depth,input_channel=32,connectivity_mode=connectivity_mode)
    add_BN_layer(net_msg,name=name+'_bn1',bottom=name+'_conv1')
    add_relu_layer(net_msg,name=name+'_relu1',bottom=name+'_bn1')
    #second conv
    add_conv_layer(net_msg,name=name+'_conv2',bottom=name+'_relu1',num_output=32,pad=1,kernel_size=3,stride=1,bias_term=False,learn_depth=learn_depth,input_channel=32,connectivity_mode=connectivity_mode)
    add_BN_layer(net_msg,name=name+'_bn2',bottom=name+'_conv2')
    #add layer
    if downsample:
        #add_downsampling_layer(net_msg,name+'_downsampling',bottom,2)
        add_conv_layer(net_msg,name=name+'_downsampling',bottom=bottom,num_output=32,pad=0,kernel_size=1,stride=2,bias_term=False,connectivity_mode=connectivity_mode)
        add_eltwise_add_layer(net_msg,name+'_add',name+'_downsampling',name+'_bn2')
    else:
        add_eltwise_add_layer(net_msg,name+'_add',bottom,name+'_bn2')
    #final relu
    add_relu_layer(net_msg,name=name,bottom=name+'_add')

def add_3rd_res_layers(net_msg,name,bottom,downsample=False,learn_depth=False,connectivity_mode=0):
    # first layer
    if downsample:
        add_conv_layer(net_msg,name=name+'_conv1',bottom=bottom,num_output=64,pad=1,kernel_size=3,stride=2,bias_term=False,learn_depth=learn_depth,input_channel=32,connectivity_mode=connectivity_mode)
    else:
        add_conv_layer(net_msg,name=name+'_conv1',bottom=bottom,num_output=64,pad=1,kernel_size=3,stride=1,bias_term=False,learn_depth=learn_depth,input_channel=64,connectivity_mode=connectivity_mode)
    add_BN_layer(net_msg,name=name+'_bn1',bottom=name+'_conv1')
    add_relu_layer(net_msg,name=name+'_relu1',bottom=name+'_bn1')
    #second conv
    add_conv_layer(net_msg,name=name+'_conv2',bottom=name+'_relu1',num_output=64,pad=1,kernel_size=3,stride=1,bias_term=False,learn_depth=learn_depth,input_channel=64,connectivity_mode=connectivity_mode)
    add_BN_layer(net_msg,name=name+'_bn2',bottom=name+'_conv2')
    #add layer
    if downsample:
        #add_downsampling_layer(net_msg,name+'_downsampling',bottom,2)
        add_conv_layer(net_msg,name=name+'_downsampling',bottom=bottom,num_output=64,pad=0,kernel_size=1,stride=2,bias_term=False,connectivity_mode=connectivity_mode)
        add_eltwise_add_layer(net_msg,name+'_add',name+'_downsampling',name+'_bn2')
    else:
        add_eltwise_add_layer(net_msg,name+'_add',bottom,name+'_bn2')

    #final relu
    add_relu_layer(net_msg,name=name,bottom=name+'_add')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_template', type=str, required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--connectivity_mode', type=int, required=True)
    #parser.add_argument('--learn_depth', type=bool, required=False)
    parser.add_argument('--learndepth', dest='learndepth', action='store_true')
    parser.add_argument('--no-learndepth', dest='learndepth', action='store_false')
    parser.set_defaults(learndepth=False)
    args = parser.parse_args()
    net_template = args.net_template
    n = args.n
    learn_depth = args.learndepth
    connectivity_mode = args.connectivity_mode

    caffe.set_mode_cpu()
    net_parser = caffeparser.CaffeProtoParser(net_template)
    net_msg = net_parser.readProtoNetFile()

    add_conv_layer(net_msg,name='conv1',bottom='data',num_output=16,pad=1,kernel_size=3,stride=1,connectivity_mode=connectivity_mode)
    add_BN_layer(net_msg,name='conv1_bn',bottom='conv1')
    add_relu_layer(net_msg,name='conv1_relu',bottom='conv1_bn')

    for i in range(1,n+1):
        if i==1:
            add_1st_res_layers(net_msg,name='res_grp1_{}'.format(i),bottom='conv1_relu',learn_depth=learn_depth,connectivity_mode=connectivity_mode)
        else:
            add_1st_res_layers(net_msg,'res_grp1_{}'.format(i),'res_grp1_{}'.format(i-1),learn_depth=learn_depth,connectivity_mode=connectivity_mode)

    for i in range(1,n+1):
        if i==1:
            add_2nd_res_layers(net_msg,name='res_grp2_{}'.format(i),bottom='res_grp1_{}'.format(n),downsample=True,learn_depth=learn_depth,connectivity_mode=connectivity_mode)
        else:
            add_2nd_res_layers(net_msg,'res_grp2_{}'.format(i),'res_grp2_{}'.format(i-1),downsample=False,learn_depth=learn_depth,connectivity_mode=connectivity_mode)

    for i in range(1,n+1):
        if i==1:
            add_3rd_res_layers(net_msg,name='res_grp3_{}'.format(i),bottom='res_grp2_{}'.format(n),downsample=True,learn_depth=learn_depth,connectivity_mode=connectivity_mode)
        else:
            add_3rd_res_layers(net_msg,'res_grp3_{}'.format(i),'res_grp3_{}'.format(i-1),downsample=False,learn_depth=learn_depth,connectivity_mode=connectivity_mode)


    #conv_cur_layer.CopyFrom(conv_layer)
    add_global_avg_pooling_layer(net_msg,name='pool1',bottom='res_grp3_{}'.format(n))
    add_ip_layer(net_msg=net_msg,name='ip1',bottom='pool1',num=10)
    add_accuracy_layer(net_msg=net_msg,bottom='ip1')
    add_loss_layer(net_msg=net_msg,bottom='ip1')

    file_split = os.path.splitext(net_template)
    filepath = 'examples/cifar10/cifar10_resnet_n{}'.format(n)+file_split[1]
    file = open(filepath, "w")
    if not file:
        raise IOError("ERROR (" + filepath + ")!")
    file.write(str(net_msg))
    file.close()

    print net_msg