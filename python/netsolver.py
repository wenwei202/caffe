# Thanks to the awesome tutorial: http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import argparse
import caffeparser
# --solver examples/cifar10/cifar10_full_multistep_solver.prototxt --weights examples/cifar10/cifar10_full_iter_50000.caffemodel.h5
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', type=str, required=True, help="Solver prototxt")
    parser.add_argument('--weights', type=str, required=False, help="Caffemodel in hdf5 format")
    args = parser.parse_args()
    solverfile = args.solver
    caffemodel = args.weights

    #caffe.set_mode_cpu()
    caffe.set_device(0)
    caffe.set_mode_gpu()

    solver = caffe.get_solver(solverfile)
    if None != caffemodel:
        solver.net.load_hdf5(caffemodel)

    solver.solve()
    #while 1:
    #    print "Move 100 steps forward..."
    #    solver.step(100)



