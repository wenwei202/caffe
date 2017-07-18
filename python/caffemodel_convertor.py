import caffe
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, required=True,help="The network prototxt")
    parser.add_argument('--caffemodel', type=str, required=True, help="The caffemodel of the network")
    args = parser.parse_args()
    network = args.network
    caffemodel = args.caffemodel

    caffe.set_mode_cpu()
    net = caffe.Net(network, caffemodel, caffe.TEST)
    file_split = os.path.splitext(caffemodel)
    if '.h5' == file_split[1]:
        net.save(file_split[0])
    elif '.caffemodel'==file_split[1] :
        net.save_hdf5(file_split[0]+file_split[1]+'.h5')
    else :
        print "File with wrong suffix"
        exit()