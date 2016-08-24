# Sparse deep neural networks with structured sparsity

This is a detached source code of [Caffe](http://caffe.berkeleyvision.org/) targeting on sparse convolutional neural networks with *structured sparisty* to speedup DNN evaluation in computation- and memory-limited devices.

Technical details are in our [NIPS 2016 paper](http://arxiv.org/abs/1608.03665).
Our *SSL (Structured Sparsity Learning)* method can dynamically learn a compact structure (less filters, less channels, smaller filter shapes and less layers) of deep neural networks, achieving speedups of AlexNet by 3.1X in GPUs and 5.1X in CPUs, using off-the-shelf GEMM in BLAS (e.g. MKL in CPUs and cuBLAS in nvidia GPUs). Alternatively, a variant of our method can improve accuracy of AlexNet by ~1%. Moreover, our results can also reduce the number of layers in Deep Residual Networks (ResNets) meanwhile improving its accuracy.

## Motivation
Deep neural networks can be very sparse (>90%), after optimization by L1 regularization or connection pruning. Although the model size can be compressed using sparsity, the computation cannot be sped up because of the poor cache locality and jumping memory access pattern.

![Alt text](/models/bvlc_reference_caffenet/speedups.png?raw=true "Speedup vs. sparsity")

The above figure shows the speedups are very limited (sometimes even slows down), althrough the sparsity can be as high as >90%. Our *SSL* method can train DNNs with structured sparsity, which results in very good locality and memory access pattern.

## Caffemodel
Uploaded in [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo#learning-structured-sparsity-in-deep-neural-networks).

## HowTo and Features
### Train sparsity-structured convolutional neural networks 
You can use our trained caffemodel in the model zoo, or train it by yourselves.

1. Stabilizing sparsity
  - Note that weights smaller than a threshold ([0.0001](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Sparse_Convolutional_Neural_2015_CVPR_paper.pdf)) are zeroed out after updating weights
2. New [caffe.proto](https://github.com/wenwei202/caffe/blob/scnn/src/caffe/proto/caffe.proto) configurations, please refer to comments in `caffe.proto` for more details
  - Training DNNs with SSL is straightforward, you only need to configure the dimensions and decays of blocks (groups) in the weight matrixes. Blocks are configured by `BlockGroupLassoSpec block_group_lasso` in each `ParamSpec` (e.g. weights). Following is an example to enable group lasso regularization on tiled 10x5 blocks in the weight matrix of conv2 layer:
  ```
  layer {
  name: "conv2"
  type: "Convolution"
  bottom: "norm1"
  top: "conv2"
  param { # weights
    lr_mult: 1
    decay_mult: 0
    # kernel_shape_decay_mult: 1 # the decay multiplier of group lasso regularization on each column
    # breadth_decay_mult: 1 # the decay multiplier of group lasso regularization on each row
    block_group_lasso { # specify the group lasso regularization on 2D blocks
      xdimen: 5 # the block size along the x (column) dimension
      ydimen: 10 # the block size along the y (row) dimension
      block_decay_mult: 1.667 # the decay multiplier of group lasso regularization on each block
    }
  }
  param { # biases
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    ... # other parameters
  }
}
  ```
  - `block_group_decay` in `SolverParameter`: do NOT forget to configure global weight decay of group lasso regularization in the solver prototxt by setting `block_group_decay` (default value is 0)
  - `connectivity_mode` in `LayerParameter` can permanently prune zero-weighted connections: if you want to freeze the zero weights in the weight matrix, please use [connectivity_mode](https://github.com/wenwei202/caffe/blob/scnn/src/caffe/proto/caffe.proto#L362).
  - local [regularization_type](https://github.com/wenwei202/caffe/blob/scnn/src/caffe/proto/caffe.proto#L316) ("L1/L2") is supported for each `ParamSpec` (e.g. weights) in each layer.


### Test/Evaluate sparsity-structured convolutional neural networks 
  - As atlas and openblas does not officially support sparse blas, please use mkl BLAS if you want to use the Compressed Sparse Row feature in CPU mode.
  - [conv_mode](https://github.com/wenwei202/caffe/blob/scnn/src/caffe/proto/caffe.proto#L637) in `ConvolutionParameter` configures the computation modality of convolution (GEMM, CSR, Concatenation, etc.). Following is an example to configure `deploy.prototxt` so that the matrix multiplication is operated by sparse weight matrix * dense feature map matrix (`LOWERED_CSRMM`), GEMM (`LOWERED_GEMM`) or Concatenation+GEMM (`LOWERED_CCNMM`): 
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
    # conv_mode: LOWERED_CCNMM # removing all-zero rows & columns and ConCateNating remaining ones, then do gemm. In GPU mode, the lowering operation is temporally implemented with CPU subroutines. 
  }
}
```


Please cite our NIPS 2016 paper and Caffe if it helps you:

    @incollection{Wen_NIPS2016,
    Title = {Learning Structured Sparsity in Deep Neural Networks},
    Author = {Wen, Wei and Wu, Chunpeng and Wang, Yandan and Chen, Yiran and Li, Hai},
    bookTitle = {Advances in Neural Information Processing Systems},
    Year = {2016}
    }
    
    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }

### Notes
Speed is compared by matrix-matrix multiplication (GEMM) in each convolutional layer, by a layer-by-layer fashion. The speedup of the total time may be different, because
  1. The implementation of lowering convolution to GEMM is not efficient in Caffe, especially in CPU mode.
  2. After the time of GEMM is squeezed, the computation time of other layers (e.g. pooling layers) comes to the surface.

However, the lowering and pooling can also be optimized by programming tricks. Please refer to our paper of [Holistic SparseCNN](https://arxiv.org/abs/1608.01409) and [intel branch](https://github.com/wenwei202/caffe/tree/intel):

    @article{park2016scnn,
      Author = {Park, Jongsoo and Li, R. Sheng and Wen, Wei and Li, Hai and Chen, Yiran and Dubey, Pradeep},
      Journal = {arXiv preprint arXiv:1608.01409},
      Title = {Holistic SparseCNN: Forging the Trident of Accuracy, Speed, and Size},
      Year = {2016}
    }

  
