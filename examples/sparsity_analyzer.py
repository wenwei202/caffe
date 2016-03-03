__author__ = 'pittnuts'
import caffe
import re
from pittnuts import *
import os
import matplotlib.pyplot as plt
import argparse
import caffeparser
# --prototxt models/bvlc_reference_caffenet/train_val.prototxt --origimodel models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel --tunedmodel /home/wew57/github/caffe_gpu0/models/bvlc_reference_caffenet/caffenet_train_grouplasso_iter_120000.caffemodel
# --prototxt examples/mnist/lenet_train_test.prototxt --origimodel examples/mnist/lenet_iter_10000.caffemodel --tunedmodel examples/mnist/lenet_grouplasso_iter_10000.caffemodel
def show_filters(net,layername):
    weights = net.params[layername][0].data
    if len(weights.shape) < 3:
        return
    weights = abs(weights)
    chan_num = weights.shape[1]

    filt_min, filt_max = weights.min(), weights.max()
    display_region_size = ceil(sqrt(chan_num))
    for n in range(min(10,weights.shape[0])):
        if sum(weights[n])>0:
            print "{}-th channel is usefull".format(n)
            plt.figure()
            for c in range(chan_num):
                plt.subplot((int)(display_region_size),(int)(display_region_size),c+1)
                #plt.title("filter #{} output".format(c))
                plt.imshow(weights[n,c], vmin=filt_min, vmax=filt_max,cmap=plt.get_cmap('Greys'),interpolation='none')
                #plt.tight_layout()
                plt.axis('off')


if __name__ == "__main__":
    # helper show filter outputs


    parser = argparse.ArgumentParser()
    parser.add_argument('--prototxt', type=str, required=True)
    parser.add_argument('--origimodel', type=str, required=True)
    parser.add_argument('--tunedmodel', type=str, required=True)
    args = parser.parse_args()
    prototxt = args.prototxt #"models/eilab_reference_sparsenet/train_val_scnn.prototxt"
    original_caffemodel = args.origimodel # "models/eilab_reference_sparsenet/eilab_reference_sparsenet.caffemodel"
    fine_tuned_caffemodel = args.tunedmodel # "/home/wew57/2bincaffe/models/eilab_reference_sparsenet/sparsenet_train_iter_30000.caffemodel"
    net_parser = caffeparser.CaffeProtoParser(prototxt)
    net_msg = net_parser.readProtoNetFile()

# --prototxt models/bvlc_reference_caffenet/train_val.prototxt  --origimodel models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel --tunedmodel models/bvlc_reference_caffenet/caffenet_train_grouplasso_iter_160000.caffemodel
# --prototxt /home/wew57/bincaffe/models/eilab_reference_sparsenet/train_val_scnn.prototxt  --origimodel /home/wew57/bincaffe/models/eilab_reference_sparsenet/eilab_reference_sparsenet_zerout.caffemodel --tunedmodel /home/wew57/bincaffe/models/eilab_reference_sparsenet/sparsenet_train_iter_160000.caffemodel
# --prototxt examples/cifar10/cifar10_full_train_test.prototxt --origimodel examples/cifar10/cifar10_full_step_iter_200000.caffemodel --tunedmodel examples/cifar10/cifar10_full_grouplasso_iter_100000.caffemodel
# --prototxt examples/mnist/lenet_train_test.prototxt --origimodel examples/mnist/lenet_iter_10000.caffemodel --tunedmodel examples/mnist/lenet_grouplasso_iter_10000.caffemodel
#    caffe.set_mode_cpu()
    # GPU mode
    caffe.set_device(1)
    caffe.set_mode_gpu()
    orig_net = caffe.Net(prototxt,original_caffemodel, caffe.TEST)
    tuned_net = caffe.Net(prototxt,fine_tuned_caffemodel, caffe.TEST)
    #orig_net = tuned_net
    print("blobs {}\nparams {}".format(orig_net.blobs.keys(), orig_net.params.keys()))
    print("blobs {}\nparams {}".format(tuned_net.blobs.keys(), tuned_net.params.keys()))
    #show_filters(tuned_net,'conv1')
    kernel_max_sizexsize = -1
    speedupinfo = ""
    plot_count = 0
    subplot_num = 0
    for layer_name in orig_net.params.keys():
        if re.match("^conv[0-9]$",layer_name):
            subplot_num += net_parser.getLayerByName(net_msg,layer_name).convolution_param.group

    r_width = 0.0001
    #einet_plt = plt.figure().add_subplot(111)
    for layer_name in orig_net.params.keys():
            if re.match("^conv.*[pq]",layer_name) :
                print "analyzing {}".format(layer_name)
                weights = orig_net.params[layer_name][0].data
                #bias = orig_net.params[layer_name][1].data
                #print "src weight {}, bias {}".format(weights.shape, bias.shape)
                #print "dst weight, bias"
                weights_orig = orig_net.params[layer_name][0].data
                weights_tuned = tuned_net.params[layer_name][0].data
#                assert (weights_orig==weights_tuned).all()
                if re.match("^fc.*",layer_name):
                    bias_orig = orig_net.params[layer_name][1].data
                    bias_tuned = tuned_net.params[layer_name][1].data
                    assert (bias_orig==bias_tuned).all()
                if re.match("^conv.*[q]",layer_name):
                    kernel_max_sizexsize = weights_tuned.shape[3]*weights_tuned.shape[2]
            elif re.match("^conv[0-9]",layer_name)  or re.match("^ip.*",layer_name) or re.match("^fc.*",layer_name):
                print "analyzing {}".format(layer_name)
                bias_orig = orig_net.params[layer_name][1].data
                bias_tuned = tuned_net.params[layer_name][1].data
                unequal_percentage = 100*sum(bias_orig!=bias_tuned)/(float)(bias_orig.size)
                #print unequal_percentage
#                assert unequal_percentage>99.0 and unequal_percentage<=100
                #assert (bias_orig!=bias_tuned).all()
                weights_orig = orig_net.params[layer_name][0].data
                weights_tuned = tuned_net.params[layer_name][0].data
                unequal_percentage = 100*sum(weights_orig!=weights_tuned)/(float)(weights_orig.size)
                #print "{}% unequal".format(unequal_percentage)
#                assert unequal_percentage>99.999 and unequal_percentage<=100

                print "[{}] original: %{} zeros".format(layer_name,100*sum((abs(weights_orig)<r_width).flatten())/(float)(weights_orig.size))
                print "[{}] tuned: %{} zeros".format(layer_name,100*sum((abs(weights_tuned)<r_width).flatten())/(float)(weights_tuned.size))
                zero_out(weights_tuned,r_width)


################################## scnn paper ##############################################
#                #analyze the average ratio of nonzero elements in S in paper SCNN
#                xIdx = range(0,kernel_max_sizexsize,1)
#                if re.match("^conv[0-9]",layer_name):
#                    nonzero_ratio = zeros((1,kernel_max_sizexsize))
#                    for i in xIdx:
#                        steps=range(i,weights_tuned.shape[1],kernel_max_sizexsize)
#                        tmp = weights_tuned[:,steps,:,:]
#                        nonzero_ratio[0,i] = sum(abs(tmp)>=r_width)/(float)(tmp.size)
#                    plt.figure(1)
#                    plt.plot(xIdx,nonzero_ratio[0,:],label=layer_name,linewidth=2.0)
#
#
#                #analyze the average ratio of nonzero elements in S in our method
#                xIdx = range(0,weights_tuned.shape[1],1)
#                if re.match("^conv[0-9]",layer_name):
#                    nonzero_ratio = zeros((1,weights_tuned.shape[1]))
#                    for i in xIdx:
#                        tmp = weights_tuned[:,i,:,:]
#                        nonzero_ratio[0,i] = sum(abs(tmp)>=r_width)/(float)(tmp.size)
#                    nonzero_ratio.sort()
#                    plt.figure(2)
#                    plot_count += 1
#                    plt.subplot(subplot_num,1,plot_count)
#                    #plt.xlabel(layer_name)
#                    plt.plot( xIdx,nonzero_ratio[0,::-1],"-r",label=layer_name,linewidth=2.0)
#                    plt.legend(loc='upper right', shadow=True)
#                    plt.axis([0, weights_tuned.shape[1],0, 1])
############################################################################################################

                #analyze the average ratio of after group lasso
                if re.match("^conv[0-9]",layer_name):
                    group = net_parser.getLayerByName(net_msg,layer_name).convolution_param.group
                    group_size = weights_tuned.shape[0]/group
                    for g in range(0,group):
                        weights_tuned_reshaped = reshape(weights_tuned[g*group_size:(g+1)*group_size,:,:,:],(weights_tuned.shape[0]/group,weights_tuned.size/weights_tuned.shape[0]))
                        nonzero_ratio = zeros((1,weights_tuned_reshaped.shape[1]))
                        xIdx = range(0,weights_tuned_reshaped.shape[1],1)
                        for i in xIdx:
                            tmp = weights_tuned_reshaped[:,i]
                            nonzero_ratio[0,i] = sum(abs(tmp)>=r_width)/(float)(tmp.size)
                        nonzero_ratio.sort()
                        plt.figure(1000)
                        plot_count += 1
                        plt.subplot(subplot_num,1,plot_count)
                        #plt.xlabel(layer_name)
                        plt.plot( xIdx,nonzero_ratio[0,::-1],"-r",label="{}_{}".format(layer_name,g),linewidth=4.0)
                        plt.legend(loc='upper right', shadow=True)
                        plt.axis([0, weights_tuned_reshaped.shape[1],0, 1])
                        show_matrix(abs(weights_tuned_reshaped)>0)
                        counts_along_row = sum(weights_tuned_reshaped!=0,axis=1)
                        col_sparsity = sum(nonzero_ratio==0)*1.0/nonzero_ratio.size
                        row_sparsity = sum(counts_along_row==0)*1.0/counts_along_row.size
                        elem_sparsity = sum(weights_tuned_reshaped==0)*1.0/weights_tuned_reshaped.size
                        plt.title("{}_{}: weight sparsity (col:{:.1%} row:{:.1%} elem:{:.1%})".format(layer_name,g,\
                                    col_sparsity, \
                                    row_sparsity,\
                                    elem_sparsity))
                        plt.xlabel("{:.2f}X speedup".format(1.0/(1-col_sparsity)/(1-row_sparsity)))
                        speedupinfo = speedupinfo + "{:.2f}X".format(1.0/(1-col_sparsity)/(1-row_sparsity))

            #else:
            #    weights_q = orig_net.params[layer_name+'q'][0].data
            #    weights_s = orig_net.params[layer_name][0].data
            #    bias_s = dst_net.params[layer_name][1].data
            #    print "P {} {}".format(weights_p.shape,0)
            #    print "Q {} {}".format(weights_q.shape,0)
            #    print "S {} {}".format(weights_s.shape,bias_s.shape)
            #    weights = transpose(weights,(2,3,1,0))
            #    group = weights_p.shape[0]/weights_p.shape[1]
            #    group_size = weights.shape[3]/group
            #    group_p_size = weights_p.shape[0]/group
            #    group_q_size = weights_q.shape[0]/group
            #    group_s_size = weights_s.shape[0]/group
            #    print "{} group(s) ".format(group)
            #    for g in range(0,group):
            #        group_weights = weights[:,:,:,g*group_size:(g+1)*group_size]
            #        P,S,Q,qi = pittnuts.kernel_factorization(group_weights)
            #        weights_p[g*group_p_size:(g+1)*group_p_size,:,0,0] = P
            #        weights_q[g*group_q_size:(g+1)*group_q_size,:,:,:] = Q.transpose((0,3,1,2)).reshape((group_q_size,weights_q.shape[1],weights_q.shape[2],weights_q.shape[3]))
            #        weights_s[g*group_s_size:(g+1)*group_s_size,:,:,:] = S.transpose(2,0,1).reshape((group_s_size,weights_s.shape[1],weights_s.shape[2],weights_s.shape[3]))
            #    bias_s[:] = bias
            #elif re.match("^fc.*",layer_name):
            #    print "filling {}".format(layer_name)
            #    dst_net.params[layer_name][0].data[:] = src_net.params[layer_name][0].data[:]
            #    dst_net.params[layer_name][1].data[:] = src_net.params[layer_name][1].data[:]

    #save zeroed out net
    file_split = os.path.splitext(fine_tuned_caffemodel)
    filepath = file_split[0]+'_zerout'+file_split[1]
    tuned_net.save(filepath)
#    plt.figure(1)
#    plt.title("avg. ratio of nonzero in S along each kernel")
#    plt.legend(loc='upper right', shadow=True)
    plt.figure(1000)
    plt.subplot(subplot_num,1,1)
    plt.title("nonzero ratio of each column (sorted)")
    plt.show()
    print speedupinfo