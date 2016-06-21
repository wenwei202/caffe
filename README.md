# Caffe for Sparse Convolutional Neural Networks

This is a fork of [Caffe](http://caffe.berkeleyvision.org/) targeting on sparse convolutional neural networks to speedup DNN evaluation in computation- and memory-limited devices.
We note that the GPU speedup utilizing random sparse weights (after L1-norm regularization or connection pruning) is very limited.

## HowTo and Features
### train sparse convolutional neural networks 
1. Stabilizing sparsity
  - weights smaller than a threshold ([0.0001](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Sparse_Convolutional_Neural_2015_CVPR_paper.pdf)) are zeroed out after update weights
2. New [caffe.proto](https://github.com/wenwei202/caffe/blob/scnn/src/caffe/proto/caffe.proto) configurations
  - *block_group_decay* in *SolverParameter*: weight decay by group lasso regularization on each group(block) of in weight matrix, block are configured by *block_group_lasso* in each *ParamSpec* (e.g. weights);
  - *connectivity_mode* in *LayerParameter* can permanently prune zero-weighted connections;
  - local *regularization_type* ("L1/L2") is supported for each *ParamSpec* (e.g. weights)

### deploy sparse convolutional neural networks 
  - [conv_mode](https://github.com/wenwei202/caffe/blob/scnn/src/caffe/proto/caffe.proto#L637) in *ConvolutionParameter* configures the computation modality of convolution (GEMM, CSR, Concatenation, etc.). Following is an example to configure deploy.prototxt so that the matrix multiplication is operated by sparse weight matrix * dense feature map matrix.
```
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "norm1"
  top: "conv2"
  convolution_param {
    num_output: 256
    pad: 2
    kernel_size: 5
    group: 2
    conv_mode: LOWERED_CSRMM # sparse weight matrix in CSR format * lowered feature maps
    # conv_mode: LOWERED_GEMM # default original matrix multiplication 
  }
}
```


Please cite Caffe and our publication if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
