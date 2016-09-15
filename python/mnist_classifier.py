'''
Main script to run classification/test/prediction/evaluation
'''
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
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--caffemodel', type=str, required=True)
    parser.add_argument('--device', type=int, required=False)
    args = parser.parse_args()
    network = args.network
    caffemodel = args.caffemodel
    device = args.device
    if device == None:
        device = 0

    val_path  = 'examples/mnist/mnist_test_lmdb/'

    if device==-1:
        caffe.set_mode_cpu()
    elif device>=0:
        # GPU mode
        caffe.set_device(device)
        caffe.set_mode_gpu()
    else :
        caffe.set_mode_cpu()

    net = caffe.Net(network,caffemodel, caffe.TEST)

    # set net to batch size
    height = 28
    width = 28
    if height!=width:
        warnings.warn("height!=width, please double check their dimension position",RuntimeWarning)

    count = 0
    correct_top1 = 0
    correct_top5 = 0
    labels_set = set()
    lmdb_env = lmdb.open(val_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    avg_time = 0
    batch_size = net.blobs['data'].num
    label = zeros((batch_size,1))
    image_count = 0
    sparsity = {}
    for blob_name in net.blobs.keys():
        sparsity[blob_name] = 0

    for key, value in lmdb_cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label[image_count%batch_size,0] = int(datum.label)
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)

        net.blobs['data'].data[image_count%batch_size] = image/float(255)
        if image_count % batch_size == (batch_size-1):
            starttime = time.time()
            out = net.forward()
            endtime = time.time()
            for blob_name in net.blobs.keys():
                sparsity[blob_name] += get_blob_sparsity(net)[blob_name]
            plabel = out['prob'][:].argmax(axis=1)
            plabel_top5 = argsort(out['prob'][:],axis=1)[:,-1:-6:-1]
            assert (plabel==plabel_top5[:,0]).all()
            count = image_count + 1
            current_test_time = endtime-starttime

            correct_top1 = correct_top1 + sum(label.flatten() == plabel.flatten())#(1 if iscorrect else 0)

            correct_top5_count = sum(contains2D(plabel_top5,label))
            correct_top5 = correct_top5 + correct_top5_count

            sys.stdout.write("\n[{}] Accuracy (Top 1): {:.2f}%".format(count,100.*correct_top1/count))
            sys.stdout.write("  (Top 5): %.2f%%" % (100.*correct_top5/count))
            sys.stdout.write("  (current time): {}\n".format(1000*current_test_time))
            sys.stdout.flush()
        image_count += 1

    print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))
    print ("Average sparsity of blobs:")
    for blob_name in net.blobs.keys():
        sparsity[blob_name] = sparsity[blob_name] / count
        print blob_name, "\t", sparsity[blob_name]