import numpy as np
import caffe
from caffe.proto import caffe_pb2

#weight_decays = 100*np.ones((20,1,5,5))
#filename="examples/mnist/conv1_decay.binaryproto"

#weight_decays = 1*np.ones((50,20,5,5))
#filename="examples/mnist/conv2_decay.binaryproto"

#weight_decays = np.zeros((500,800))
#filename="examples/mnist/ip1_decay.binaryproto"

weight_decays = np.zeros((10,500))
filename="examples/mnist/ip2_decay.binaryproto"


blobproto = caffe.proto.caffe_pb2.BlobProto()
blobproto = caffe.io.array_to_blobproto(weight_decays)

f = open(filename, "wb")
f.write(blobproto.SerializeToString())
f.close()