# Caffe for Sparse Convolutional Neural Networks

This is a fork of Caffe targeting on sparse convolutional neural networks to speedup DNN evaluation in computation- and memory-limited devices.


## HowTo and Features
### train sparse convolutional neural networks 
1. weights smaller than a threshold ([0.0001](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Sparse_Convolutional_Neural_2015_CVPR_paper.pdf)) are zeroed out after each weight updating to stabilize sparsity;
2. new [caffe.proto](https://github.com/wenwei202/caffe/blob/master/src/caffe/proto/caffe.proto) configurations:
    2.1. *block_group_decay* in *SolverParameter*: weight decay by group lasso regularization on each group(block) of in weight matrix, block are configured by *block_group_lasso* in each *ParamSpec* (e.g. weights);
    2.2. *connectivity_mode* in *LayerParameter* can permanently prune zero-weighted connections;
    2.3. local *regularization_type* ("L1/L2") is supported for each *ParamSpec* (e.g. weights)

### deploy sparse convolutional neural networks 
1. *conv_mode* in *ConvolutionParameter* configures the computation modality of convolution (GEMM, CSR, Concatenation, etc.)

Please cite Caffe in your publications if it helps your research.
