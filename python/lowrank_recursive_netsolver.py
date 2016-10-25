import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import argparse
import caffeparser
import re
import caffe_apps
import os
import copy
import gc



def lowrank_netsolver(solverfile,caffemodel,ratio,rank_mat):
    solver_parser = caffeparser.CaffeProtoParser(solverfile)
    solver_msg = solver_parser.readProtoSolverFile()
    lr_policy = str(solver_msg.lr_policy)
    if lr_policy!="multistep":
        print "Only multistep lr_policy is supported in our lowrank_netsolver!"
        exit()
    max_iter = solver_msg.max_iter
    test_interval = solver_msg.test_interval
    stepvalues = copy.deepcopy(solver_msg.stepvalue)
    #stepvalues.append(max_iter)
    base_lr = solver_msg.base_lr

    net_parser = caffeparser.CaffeProtoParser(str(solver_msg.net))
    net_msg = net_parser.readProtoNetFile()
    loop_layers = net_msg.layer

    solver = caffe.get_solver(solverfile)
    if None != caffemodel:
        solver.net.load_hdf5(caffemodel)
    iter = 0
    filepath_solver=""
    filepath_caffemodel=""
    while iter<max_iter:
        # train for some steps
        solver.step(test_interval)

        # initialize the parameters in the new network
        new_parameters = {}
        for cur_layer in loop_layers:
            if cur_layer.name in solver.net.params:
                cur_param = {}
                for idx in range(0, len(solver.net.params[cur_layer.name])):
                    cur_param[idx] = solver.net.params[cur_layer.name][idx].data[:]
                new_parameters[cur_layer.name] = cur_param

        # check if a lower rank in each layer can be obtained
        # if so, update the network structure and weights
        layer_idx = -1
        new_net_flag = False
        rank_info = ""
        ranks = [[]]
        for cur_layer in loop_layers:
            layer_idx += 1
            if 'Convolution' == cur_layer.type and re.match(".*(_lowrank)$", cur_layer.name):
                assert len(solver.net.params[cur_layer.name]) == 1
                cur_weights = solver.net.params[cur_layer.name][0].data
                next_layer = net_msg.layer._values[layer_idx + 1]
                next_weights = solver.net.params[next_layer.name][0].data
                assert re.match(".*(_linear)$",next_layer.name)
                assert len(solver.net.params[next_layer.name]) == 2
                assert next_layer.convolution_param.kernel_size._values[0] == 1
                low_rank_filters, linear_combinations, rank = caffe_apps.filter_pca(cur_weights, ratio)
                rank_info = rank_info + "_{}".format(rank)
                ranks[0].append(rank)
                if rank < cur_weights.shape[0]: # generate lower-rank network
                    new_net_flag = True
                    cur_layer.convolution_param.num_output = rank
                    new_parameters[cur_layer.name] = {0: low_rank_filters[:]}
                    new_linear_combinations = np.dot(next_weights.reshape((next_weights.shape[0],-1)), linear_combinations.reshape((linear_combinations.shape[0],-1)))
                    new_linear_combinations = new_linear_combinations.reshape((next_layer.convolution_param.num_output,rank,1,1))
                    if  next_layer.convolution_param.bias_term:
                        new_parameters[next_layer.name] = {0: new_linear_combinations[:],
                                                             1: solver.net.params[next_layer.name][1].data[:]}
                    else:
                        new_parameters[next_layer.name] = {0: new_linear_combinations[:]}

        iter += test_interval
        if []==rank_mat:
            rank_mat=ranks
        else:
            rank_mat = np.concatenate((rank_mat,ranks),axis=0)

        # snapshot network, caffemodel and solver
        if new_net_flag:
            # save the new network
            #file_split = os.path.splitext(str(solver_msg.net))
            filepath_network = solver_msg.snapshot_prefix+rank_info+"_net.prototxt" #file_split[0] + '_lowrank' + file_split[1]
            file = open(filepath_network, "w")
            if not file:
                raise IOError("ERROR (" + filepath_network + ")!")
            file.write(str(net_msg))
            file.close()
            print "Saved as {}".format(filepath_network)

            # save new soler
            solver_msg.net = filepath_network
            next_lr = base_lr
            left_steps = copy.deepcopy(stepvalues)
            for idx, step_val in enumerate(stepvalues):
                if iter >= step_val:
                    next_lr = next_lr * solver_msg.gamma
                left_steps[idx] = step_val - iter
            solver_msg.base_lr = next_lr
            solver_msg.max_iter = max_iter - iter

            if -1!=solver_msg.force_iter and 0!=solver_msg.force_iter:
                solver_msg.force_iter = solver_msg.force_iter - iter
                if solver_msg.force_iter < 0:
                    solver_msg.force_iter = 0

            solver_msg.stepvalue._values=[]
            for idx, step_val in enumerate(left_steps):
                if step_val > 0:
                    solver_msg.stepvalue.append(step_val)
            filepath_solver = solver_msg.snapshot_prefix + rank_info + "_solver.prototxt"  # file_split[0] + '_lowrank' + file_split[1]
            file = open(filepath_solver, "w")
            if not file:
                raise IOError("ERROR (" + filepath_solver + ")!")
            file.write(str(solver_msg))
            file.close()
            print "Saved as {}".format(filepath_solver)

            # generate the caffemodel
            if iter == max_iter:
                solver.solve()
            solver = None # a weird bug if do not release it
            gc.collect()
            dst_net = caffe.Net(str(filepath_network), caffe.TRAIN)
            for key, val in new_parameters.iteritems():
                for keykey, valval in val.iteritems():
                    dst_net.params[key][keykey].data[:] = valval[:]
            filepath_caffemodel = solver_msg.snapshot_prefix + rank_info+".caffemodel.h5"
            dst_net.save_hdf5(str(filepath_caffemodel))
            print "Saved as {}".format(filepath_caffemodel)
            dst_net = None # a weird bug if do not release it
            gc.collect()

            break


    if iter >= max_iter:
        if solver!=None :
            solver.solve()
        print "Optimization done!"
        plt.plot(rank_mat)
        plt.savefig(str(solver_msg.snapshot_prefix)+"_ranks.png")
        np.savetxt(str(solver_msg.snapshot_prefix)+".ranks",rank_mat,fmt="%d")
        #plt.show()
        return
    else :
        lowrank_netsolver(str(filepath_solver),str(filepath_caffemodel),ratio,rank_mat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', type=str, required=True, help="Solver prototxt")
    parser.add_argument('--weights', type=str, required=False, help="Caffemodel in hdf5 format")
    parser.add_argument('--ratio', type=float, required=False, help="The ratio of reserved info after pca")
    parser.add_argument('--device', type=int, required=False,help="The GPU device id, -1 for CPU")
    args = parser.parse_args()
    solverfile = args.solver
    caffemodel = args.weights
    file_split = os.path.splitext(caffemodel)
    assert ".h5" == file_split[1]

    ratio = args.ratio
    if ratio == None:
        ratio = 0.99

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
    rank_mat = []
    lowrank_netsolver(solverfile,caffemodel,ratio,rank_mat)
