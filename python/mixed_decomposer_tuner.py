import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import argparse
import caffeparser
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Low-rank approximating an fine-tuning")
    parser.add_argument('--solver', type=str, required=True, help="Solver prototxt")
    parser.add_argument('--model', required=True, help="Prototxt of the original net")
    parser.add_argument('--weights', type=str, required=True, help="Caffemodel in hdf5 format")
    parser.add_argument('--rank_config1', required=True, help="JSON config file 1 specifying the low-rank approximation")
    parser.add_argument('--rank_config2', required=True, help="JSON config file 2 specifying the low-rank approximation")
    parser.add_argument('--device', type=int, required=False,help="The GPU device id, -1 for CPU")
    parser.add_argument('--path', type=str, required=True, help="The path to store generated net model and weight caffemodel")
    args = parser.parse_args()
    save_model = args.path + "/lowrank_net.prototxt"
    save_weights = args.path + "/lowrank_weights.caffemodel.h5"
    # e.g.
    # python/lowrank_approx.py \
    # --config models/bvlc_alexnet/config_iclr.json \
    # --model models/bvlc_alexnet/train_val.prototxt \
    # --weights models/bvlc_alexnet/bvlc_alexnet.caffemodel.h5
    script_str1 = "python python/lowrank_approx.py " + \
                  " --config " + args.rank_config1 + \
                  " --model " + args.model + \
                  " --weights " + args.weights + \
                  " --save_model " + save_model + \
                  " --save_weights " + save_weights
    os.system(script_str1)

    # e.g. python python/nn_decomposer.py \
    # --prototxt models/bvlc_alexnet/train_val.prototxt \
    # --caffemodel models/bvlc_alexnet/bvlc_alexnet.caffemodel.h5 \
    # --rank_config models/bvlc_alexnet/config.json
    script_str2 = "python python/nn_decomposer.py " + \
                  " --prototxt " + save_model + \
                  " --caffemodel " + save_weights + \
                  " --rank_config " + args.rank_config2
    os.system(script_str2)
    filepath_network = save_model + ".lowrank.prototxt"
    filepath_caffemodel = save_weights + '.lowrank.caffemodel.h5'

    # e.g. python python/netsolver.py \
    # --solver models/bvlc_alexnet/solver.prototxt \
    # --weights models/bvlc_alexnet/bvlc_alexnet.caffemodel.h5 \
    # --device 0
    solver_parser = caffeparser.CaffeProtoParser(args.solver)
    solver_msg = solver_parser.readProtoSolverFile()
    solver_msg.net = filepath_network
    file = open(args.solver, "w")
    if not file:
        raise IOError("ERROR (" + args.solver + ")!")
    file.write(str(solver_msg))
    file.close()

    script_str3 = "python  python/netsolver.py " + \
                  " --device {}".format(args.device)  + \
                  " --solver " + args.solver + \
                  " --weights " + filepath_caffemodel
    os.system(script_str3)
