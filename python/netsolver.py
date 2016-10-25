import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import argparse
import os
# --solver examples/cifar10/cifar10_full_multistep_solver.prototxt --weights examples/cifar10/cifar10_full_iter_50000.caffemodel.h5
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', type=str, required=True, help="Solver prototxt")
    parser.add_argument('--weights', type=str, required=False, help="Caffemodel in hdf5 format")
    parser.add_argument('--device', type=int, required=False,help="The GPU device id, -1 for CPU")
    args = parser.parse_args()
    solverfile = args.solver
    caffemodel = args.weights
    file_split = os.path.splitext(caffemodel)
    assert ".h5" == file_split[1]

    device = args.device
    if device == None:
        device = 0

    if device == -1:
        caffe.set_mode_cpu()
    elif device >= 0:
        # GPU mode
        caffe.set_device(device)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    solver = caffe.get_solver(solverfile)
    if None != caffemodel:
        solver.net.load_hdf5(caffemodel)

    solver.solve()
    #while 1:
    #    print "Move 100 steps forward..."
    #    solver.step(100)



