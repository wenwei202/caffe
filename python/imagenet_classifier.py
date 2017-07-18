__author__ = 'pittnuts'
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

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

imagenet_val_path  = 'examples/imagenet/ilsvrc12_val_lmdb'
mean_file = 'data/ilsvrc12/imagenet_mean.binaryproto'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Low-rank approximating an fine-tuning")
    parser.add_argument('--model', required=True, help="Prototxt of the original net")
    parser.add_argument('--weights', type=str, required=True, help="Caffemodel in hdf5 format")
    parser.add_argument('--device', type=int, required=True,help="The GPU device id, -1 for CPU")
    args = parser.parse_args()

    if -1 == args.device:
        caffe.set_mode_cpu()
    else:
        # GPU mode
        caffe.set_device(args.device)
        caffe.set_mode_gpu()


    net = caffe.Net(args.model,args.weights,
                 caffe.TEST)

    # set net to batch size
    height = 227
    width = 227
    if height!=width:
        warnings.warn("height!=width, please double check their dimension position",RuntimeWarning)


    count = 0
    correct_top1 = 0
    correct_top5 = 0
    labels_set = set()
    lmdb_env = lmdb.open(imagenet_val_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    mean_blob = caffe.proto.caffe_pb2.BlobProto()
    mean_data = open( mean_file , 'rb' ).read()
    mean_blob.ParseFromString(mean_data)
    pixel_mean = np.array( caffe.io.blobproto_to_array(mean_blob) )

    avg_time = 0
    net.blobs['data'].reshape(1,3,height,width)
    batch_size = net.blobs['data'].num
    label = zeros((batch_size,1))
    image_count = 0
    for key, value in lmdb_cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label[image_count%batch_size,0] = int(datum.label)
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)

        image = image-pixel_mean.mean(0)
        image = image[:,14:14+227,14:14+227]
        net.blobs['data'].data[image_count%batch_size] = image#image-pixel_mean
        if image_count % batch_size == (batch_size-1):
            starttime = time.time()
            out = net.forward()
            endtime = time.time()
            # save blobs
            if image_count<5:
                blob_cells = {}
                for blob_name in net.blobs.keys():
                    blob_cells[blob_name] = net.blobs[blob_name].data
                savemat('blobs{}.mat'.format(image_count),blob_cells)
            plabel = out['prob'][:].argmax(axis=1)
            plabel_top5 = argsort(out['prob'][:],axis=1)[:,-1:-6:-1]
            assert (plabel==plabel_top5[:,0]).all()
            count = image_count + 1
            current_test_time = endtime-starttime

            #iscorrect = label == plabel
            correct_top1 = correct_top1 + sum(label.flatten() == plabel.flatten())#(1 if iscorrect else 0)

            #iscorrect_top5 = contains(plabel_top5,label)
            correct_top5_count = sum(contains2D(plabel_top5,label))
            correct_top5 = correct_top5 + correct_top5_count

            sys.stdout.write("\n[{}] Accuracy (Top 1): {:.6f}%".format(count,100.*correct_top1/count))
            sys.stdout.write("  (Top 5): %.6f%%" % (100.*correct_top5/count))
            sys.stdout.write("  (current time): {}\n".format(1000*current_test_time))
            sys.stdout.flush()
        image_count += 1

    plt.show()
