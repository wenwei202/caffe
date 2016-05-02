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
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
caffe_root = './'

val_path  = 'examples/cifar10/cifar10_test_lmdb'

# GPU mode
#caffe.set_device(0)
#caffe.set_mode_gpu()

caffe.set_mode_cpu()

net = caffe.Net(caffe_root + 'examples/cifar10/cifar10_full_ccnmm.prototxt',
              caffe_root + 'examples/cifar10/0.001_0.0_0.0_0.0_0.001_Thu_Apr__7_14-51-10_EDT_2016/cifar10_full_iter_200000.caffemodel',
              caffe.TEST)

# set net to batch size
height = 32
width = 32
if height!=width:
    warnings.warn("height!=width, please double check their dimension position",RuntimeWarning)

count = 0
correct_top1 = 0
correct_top5 = 0
labels_set = set()
lmdb_env = lmdb.open(val_path)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

#pixel_mean = np.load(caffe_root + 'examples/cifar10/mean.binaryproto').mean(1).mean(1)
mean_blob = caffe.proto.caffe_pb2.BlobProto()
mean_data = open( 'examples/cifar10/mean.binaryproto' , 'rb' ).read()
mean_blob.ParseFromString(mean_data)
#pixel_mean = np.array( caffe.io.blobproto_to_array(mean_blob) ).mean(2).mean(2)
#pixel_mean = tile(pixel_mean.reshape([1,3]),(height*width,1)).reshape(height,width,3).transpose(2,0,1)

pixel_mean = np.array( caffe.io.blobproto_to_array(mean_blob) ).mean(0)

avg_time = 0
batch_size = net.blobs['data'].num
label = zeros((batch_size,1))
image_count = 0
for key, value in lmdb_cursor:
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
    label[image_count%batch_size,0] = int(datum.label)
    image = caffe.io.datum_to_array(datum)
    image = image.astype(np.uint8)

    #out = net.forward_all(data=np.asarray([image]))
    #image_tmp = image[(0,1,2),:,:]
    #image_tmp = image_tmp.transpose(0,2,1)
    #plt.imshow(image.transpose(1,2,0)[:,:,(2,1,0)])
    #plt.show()

    #crop_range = range(14,14+227)
    #image = image[:,14:14+227,14:14+227]

    net.blobs['data'].data[image_count%batch_size] = image-pixel_mean
    if image_count % batch_size == (batch_size-1):
        starttime = time.time()
        out = net.forward()
        endtime = time.time()
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

plt.show()