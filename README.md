# Sparse deep neural networks with structured sparsity

This is a detached source code of [Caffe](http://caffe.berkeleyvision.org/) targeting on sparse deep neural networks with *structured sparisty* to speedup the evaluation of Deep Neural Networks (DNN).

Technical details are in our [NIPS 2016 paper](http://papers.nips.cc/paper/6504-learning-structured-sparsity-in-deep-neural-networks.pdf): **Learning Structured Sparsity in Deep Neural Networks**.
Our *SSL (Structured Sparsity Learning)* method utilizes group Lasso regularization to dynamically learn a compact DNN structure (less filters, less channels,smaller filter shapes, less neurons and less layers), achieving speedups of convolutional layers in AlexNet by 3.1X in GPUs and 5.1X in CPUs, measured by the off-the-shelf GEMM in BLAS (e.g. MKL in CPUs and cuBLAS in nvidia GPUs). Alternatively, a variant of our method can improve accuracy of AlexNet by ~1%. Moreover, our results can also reduce the number of layers in Deep Residual Networks (ResNets) meanwhile improving its accuracy.

Slides are [here](/docs/WEN_NIPS2016.pdf).

## Motivation
Deep neural networks can be very sparse (>90%), after optimization by L1 regularization or connection pruning. The model size can be compressed using those sparsifying methods, however, the computation cannot be sped up because of the poor cache locality and jumping memory access pattern resulted from the random pattern of the sparsity.

![Alt text](/models/bvlc_reference_caffenet/speedups.png?raw=true "Speedup vs. sparsity")

The above figure shows the speedups are very limited (sometimes even slows down) althrough the sparsity can be as high as >90%. Our *SSL* method can train DNNs with structured sparsity, which results in very good locality and memory access pattern. For example, SSL can directly reduce the dimensions of weight matrixes in both convolutional layers and fully-connected layers.

## Caffemodel and examples
Caffemodels of AlexNet learned by SSL are uploaded to [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo#learning-structured-sparsity-in-deep-neural-networks).

Please check [examples/cifar10](/examples/cifar10/readme.md) for the detailed tutorial to use the code.

## Issue
Let us know here if any question: [Caffe Issue 4328](https://github.com/BVLC/caffe/issues/4328)

## Overview of Features
Please refer to our [NIPS 2016 paper](http://arxiv.org/abs/1608.03665) for technical details.
### Train sparsity-structured deep neural networks 
You can use our trained caffemodel in the model zoo, or train it by yourselves.
In brief, SSL enforces group Lasso regularization on every sub-block (e.g. row, column or 2D tile) of the weight matrix. After SSL, a big portion of sub-blocks will be enforced to all-zeros. Note that the *block* and *group* are interchangeable in this context.

Training by SSL is simple, we add new features in [caffe.proto](/src/caffe/proto/caffe.proto), in which the group Lasso regularization, connectivity, implementations of convolution and so on can be configured in prototxt of networks and solvers. Please refer to comments in `caffe.proto` for more details.
  - Following is an example to configure the dimensions of sub-blocks and the coefficients of weight decay by group Lasso regularization. Sub-blocks are configured by `BlockGroupLassoSpec block_group_lasso` in each `ParamSpec param` (e.g. weights). Following is an example to enable group lasso regularization on 10x5 sub-blocks evenly tiled across the weight matrix of conv2 layer:
```
  layer {
  name: "conv2"
  type: "Convolution"
  bottom: "norm1"
  top: "conv2"
  param { # weights
    lr_mult: 1
    decay_mult: 0
    block_group_lasso { # specify the group lasso regularization on 2D blocks
      xdimen: 5 # the block size along the x (column) dimension
      ydimen: 10 # the block size along the y (row) dimension
      block_decay_mult: 1.667 # the local multiplier of weight decay by group lasso regularization
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
  - Group Lasso regularization on each row or column can be specified by `block_group_lasso`. However, we also implemented (`kernel_shape_decay_mult` & `breadth_decay_mult` in `ParamSpec param`) and (`kernel_shape_decay` & `breadth_decay`  in `SolverParameter`) to configure the group Lasso regularization on row and column respectively. 
  - `connectivity_mode` in `LayerParameter` can permanently prune zero-weighted connections: if you want to freeze the zero weights in the weight matrix, please use [connectivity_mode](/src/caffe/proto/caffe.proto#L362).
  - local [regularization_type](/src/caffe/proto/caffe.proto#L316) ("L1/L2") is supported for each `ParamSpec` (e.g. weights) in each layer.


### Test/Evaluate sparsity-structured deep neural networks 
Testing the SSL learned DNN is also simple by configuring prototxt (i.e. `conv_mode`). Note that ATLAS and OpenBLAS do not officially support sparse BLAS, please use mkl BLAS if you want to use the Compressed Sparse Row feature in CPU mode.
  - [conv_mode](/src/caffe/proto/caffe.proto#L637) in `ConvolutionParameter` configures the implementations of convolution (GEMM, CSR or Concatenation+GEMM). Following is an example to configure `deploy.prototxt` so that the matrix multiplication is operated by (sparse weight matrix) * (dense feature map matrix) (`LOWERED_CSRMM`), GEMM (`LOWERED_GEMM`) or Concatenation+GEMM (`LOWERED_CCNMM`): 
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
 - Open `USE_PROFILE_DISPLAY := 1` in `Makefile.config`. Configure the paths of database, network and caffemodel  in [examples/caffenet_classifier.py](/examples/caffenet_classifier.py) and run it to start profile. Make sure you correctly configured the `conv_mode` in each convolutional layer in the `deploy.prototxt`. Then, you will get the profiling results showing which implementation of convolution is using and what is the computation time for each layer, similar to:
 - 
 ```
 I0907 14:37:47.134873 26836 base_conv_layer.cpp:651] conv2	 group 0: 320 us (Dense Scheme Timing)
 ```
 
  - `Dense Scheme Timing` -> `LOWERED_GEMM`
  - `Compressed Row Storage Timing` -> `LOWERED_CSRMM`
  - `Concatenation Timing` -> `LOWERED_CCNMM`


Note that the weight matrixes of convolutional and fully-connected layers are snapshotted as `$CAFFE_ROOT/layername.weight`, the format obeys [Matrix Market](http://math.nist.gov/MatrixMarket/). You can use interfaces of [C](http://math.nist.gov/MatrixMarket/mmio-c.html), Fortran and [Matlab](http://math.nist.gov/MatrixMarket/mmio/matlab/mmiomatlab.html) to read those weight matrixes.

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

## Notes
1. Stabilizing sparsity
  - Note that weights smaller than a threshold ([0.0001](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Sparse_Convolutional_Neural_2015_CVPR_paper.pdf)) are zeroed out after updating weights
  
2. Caffe version
  - scnn branch is forked from caffe @ commit [eb4ba30](https://github.com/BVLC/caffe/tree/eb4ba30e3c4899edc7a9713158d61503fa8ecf90)
3. Speed is compared by matrix-matrix multiplication (GEMM) in each convolutional layer (by MKL BLAS in CPU mode and cuBLAS (not cuDNN) in GPU mode), in a layer-by-layer fashion. The speedup of the total time may be different, because
  1. The implementation of lowering convolution to GEMM is not efficient in Caffe, especially in CPU mode.
  2. After the time of GEMM is squeezed, the computation time of other layers (e.g. pooling layers) comes to the surface.
4. In GPU mode, the lowering operation to shrink feature matrix is temporally implemented with CPU subroutines. Please pull request if you implemented it in GPU mode.

However, the lowering and pooling can also be optimized by programming tricks. Please refer to our paper of [Holistic SparseCNN](https://arxiv.org/abs/1608.01409) and [intel branch](https://github.com/wenwei202/caffe/tree/intel):

    @article{park2016scnn,
      Author = {Park, Jongsoo and Li, R. Sheng and Wen, Wei and Li, Hai and Chen, Yiran and Dubey, Pradeep},
      Journal = {arXiv preprint arXiv:1608.01409},
      Title = {Holistic SparseCNN: Forging the Trident of Accuracy, Speed, and Size},
      Year = {2016}
    }

### Issues
1. `make runtest`: see reports [here](https://github.com/BVLC/caffe/issues/4328#issuecomment-229263764)
2. cudnn: version of cudnn 5 is supported
3. More in [Caffe Issue 4328](https://github.com/BVLC/caffe/issues/4328)
4. To profile, use `deploy.prototxt` in Python and use `train_val.prototxt` in `caffe time ...`, otherwise, the there might be some bugs in original Caffe. Note that training using `LOWERED_CSRMM` or `LOWERED_CCNMM` is forbidden. `caffe time` calls backward function of each layer, to use `caffe time` to profile, comment backward related codes in `tools/caffe.cpp:time()` function.

