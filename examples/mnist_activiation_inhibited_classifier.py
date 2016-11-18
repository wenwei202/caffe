__author__ = 'pittnuts'

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import *
from PIL import Image
import caffe
import sys
import lmdb
from caffe.proto import caffe_pb2
from pittnuts import *
from os import system
from caffe_apps import *
import argparse
import bottleneck as bn

# helper show filter outputs
def show_filter_outputs(net,blobname):
    if len(net.blobs[blobname].data.shape) < 3:
        return
    feature_map_num = net.blobs[blobname].data.shape[1]
    plt.figure()
    filt_min, filt_max = net.blobs[blobname].data.min(), net.blobs[blobname].data.max()
    display_region_size = ceil(sqrt(feature_map_num))
    for i in range(feature_map_num):
        plt.subplot((int)(display_region_size),(int)(display_region_size),i+1)
        #plt.title("filter #{} output".format(i))
        plt.imshow(net.blobs[blobname].data[0,i], vmin=filt_min, vmax=filt_max)
        #plt.tight_layout()
        plt.axis('off')

val_path="examples/mnist/mnist_test_lmdb"

#--imagenet_val_path examples/imagenet/ilsvrc12_val_lmdb --prototxt models/eilab_reference_sparsenet/deploy_scnn.prototxt --caffemodel models/eilab_reference_sparsenet/eilab_reference_sparsenet_zerout.caffemodel
#--imagenet_val_path examples/imagenet/ilsvrc12_val_lmdb --prototxt models/bvlc_reference_caffenet/deploy.prototxt --caffemodel models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prototxt', type=str, required=True)
    parser.add_argument('--caffemodel', type=str, required=True)
    parser.add_argument('--threshold', type=float, required=False)
    parser.set_defaults(threshold=0.0)
    args = parser.parse_args()
    prototxt=args.prototxt
    caffedmodel=args.caffemodel

    caffe.set_device(1)
    caffe.set_mode_gpu()

    net = caffe.Net( prototxt, caffedmodel, caffe.TEST)
    print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))
    height = 28
    width = 28
    if height!=width:
        warnings.warn("height!=width, please double check their dimension position",RuntimeWarning)

    net.blobs['data'].reshape(1,1,height,width)
    count = 0
    correct_top1 = 0
    correct_top5 = 0
    labels_set = set()
    lmdb_env = lmdb.open(val_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    layers = net.layers
    #layer_prop_from=('conv1','conv2','conv3','conv4','conv5')
    #layer_prop_to=('norm1','norm2','relu3','relu4','prob')
    #layer_prop_from=('conv1','conv4','conv5')
    #layer_prop_to=('relu3','relu4','prob')
    layer_prop_from=('ip1','ip2','ip3')
    layer_prop_to=('relu1','relu2','prob')
    average_sparsity = zeros((1,len(layer_prop_from)))

    for key, value in lmdb_cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
        net.blobs['data'].data[0] = image / float(255)
        for prop_step in range(0,len(layer_prop_from)):
            end_layername = layer_prop_to[prop_step]
            #print end_layername
            out = net.forward(start=layer_prop_from[prop_step],end=end_layername)
            tmp_out = abs(out[end_layername]).flatten()
            thre = args.threshold# 0#tmp_out[argsort(tmp_out)[round(tmp_out.size*40/100)]]
            if prop_step!=len(layer_prop_from)-1:
                zero_out(net.blobs[end_layername].data,thre)
                average_sparsity[0,prop_step] = (average_sparsity[0,prop_step]*count + get_sparsity(net.blobs[end_layername].data,0.0001))/(count+1)
                if count<10:
                    thefile = open(layer_prop_from[prop_step+1] + ".feature{}".format(count) + ".txt", 'w')
                    for ndx, iterm_ in enumerate(net.blobs[end_layername].data[:].flatten()):
                        thefile.write("%.6f\n" % (float)(iterm_))
                    thefile.close()
                else:
                    exit()

        plabel = int(out['prob'][0].argmax(axis=0))
        plabel_top5 = argsort(out['prob'][0])[-1:-6:-1]
        assert plabel==plabel_top5[0]
        count = count + 1

        iscorrect = label == plabel
        correct_top1 = correct_top1 + (1 if iscorrect else 0)

        iscorrect_top5 = contains(plabel_top5,label)
        correct_top5 = correct_top5 + (1 if iscorrect_top5 else 0)

        labels_set.update([label, plabel])

        sys.stdout.write("\n[{}] Accuracy (Top 1): {:.1f}%".format(count,100.*correct_top1/count))
        sys.stdout.write("  (Top 5): %.2f%%" % (100.*correct_top5/count))
        sys.stdout.write("  (sparsity): " + array_str(average_sparsity))
        sys.stdout.flush()