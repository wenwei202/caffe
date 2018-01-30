# Sparse deep neural networks with structured sparsity

This is a detached source code of [Caffe](http://caffe.berkeleyvision.org/) targeting on sparse deep neural networks with *structured sparisty* to speedup the evaluation of Deep Neural Networks (DNN).

Technical details are in our [NIPS 2016 paper](http://papers.nips.cc/paper/6504-learning-structured-sparsity-in-deep-neural-networks.pdf): **Learning Structured Sparsity in Deep Neural Networks**.
Our *SSL (Structured Sparsity Learning)* method utilizes group Lasso regularization to dynamically learn a compact DNN structure (less filters, less channels,smaller filter shapes, less neurons and less layers), achieving speedups of convolutional layers in AlexNet by 3.1X in GPUs and 5.1X in CPUs, measured by the off-the-shelf GEMM in BLAS (e.g. MKL in CPUs and cuBLAS in nvidia GPUs). Alternatively, a variant of our method can improve accuracy of AlexNet by ~1%. Moreover, our results can also reduce the number of layers in Deep Residual Networks (ResNets) meanwhile improving its accuracy.

Poster is [here](/docs/Poster_Wen_NIPS2016.pdf).
Slides are [here](/docs/WEN_NIPS2016.pdf).

This work is extended and advanced to Recurrent Neural Networks (like LSTMs and Recurrent Highway Networks) to learn hidden sizes. Related [paper](https://arxiv.org/abs/1709.05027) is publised in ICLR 2018. [Code](https://github.com/wenwei202/iss-rnns/) is implemented by TensorFlow. 

Our ICCV 2017 paper on lower-rank DNNs ([Paper](https://arxiv.org/abs/1703.09746), [Code](https://github.com/wenwei202/caffe)) can combine with SSL method for further acceleration. 

You may have interest in our NIPS 2017 oral paper, regarding acceleration of distributed training using ternary gradients:
[Paper](https://arxiv.org/abs/1705.07878), [Code](https://github.com/wenwei202/terngrad). 

## Motivation
Deep neural networks can be very sparse (>90%), after optimization by L1 regularization or connection pruning. The model size can be compressed using those sparsifying methods, however, the computation cannot be sped up because of the poor cache locality and jumping memory access pattern resulted from the random pattern of the sparsity.

![Alt text](/models/bvlc_reference_caffenet/speedups.png?raw=true "Speedup vs. sparsity")

The above figure shows the speedups are very limited (sometimes even slows down) althrough the sparsity can be as high as >90%. Our *SSL* method can train DNNs with structured sparsity, which results in very good locality and memory access pattern. For example, SSL can directly reduce the dimensions of weight matrixes in both convolutional layers and fully-connected layers.

## Build 
To build, please follow the standard process of building caffe.
New configurations are added in `Makefile` and `Makefile.config.example`, but those features are Not modified for `cmake`. So `cmake` is NOT support currently.

## Caffemodel and examples
Caffemodels of **CaffeNet** (a variant of AlexNet) learned by SSL are uploaded to [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo#learning-structured-sparsity-in-deep-neural-networks). Note that the paper focuses on acceleration and those caffemodels are NOT compressed by removing zeros. Zeros are stored as nonzeros are. During testing, zeros are removed and compressed at the beginning by `Layer::WeightAlign()` of [conv layer](https://github.com/wenwei202/caffe/blob/scnn/src/caffe/layers/base_conv_layer.cpp#L13) and [fully-connected layer](https://github.com/wenwei202/caffe/blob/scnn/src/caffe/layers/inner_product_layer.cpp#L10).

Please check examples in [examples/cifar10](/examples/cifar10/readme.md) and [models/vggnet](/models/vggnet/).

## Issue
Submit an issue [here](https://github.com/wenwei202/caffe/issues).

Or, refer to [Caffe Issue 4328](https://github.com/BVLC/caffe/issues/4328)

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
      block_decay_mult: 1.667 # the local multiplier of weight decay (block_group_decay) by group lasso regularization
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
  - Group Lasso regularization on each row or column can be specified by `block_group_lasso` with `ydimen: 1` or `xdimen: 1`. However, in GPU mode, we also implemented (`breadth_decay_mult` & `kernel_shape_decay_mult` in `ParamSpec param`) and (`breadth_decay` & `kernel_shape_decay` in `SolverParameter`) to simplify the configuration of group Lasso regularization on each row or column, respectively. For example, in `conv1` of LeNet, `kernel_shape_decay_mult: 1.5` is equivalent to 
  
  ```
  param { # weights
    lr_mult: 1
    block_group_lasso { # specify the group lasso regularization each column
      xdimen: 1 
      ydimen: 20 # The size of each column is the number of filters 
      block_decay_mult: 1.5 # the same with kernel_shape_decay_mult
    }
  }
  ```
  and `breadth_decay_mult: 1.5` is equivalent to
  
  ```
  param { # weights
    lr_mult: 1
    block_group_lasso { # specify the group lasso regularization each row
      xdimen: 75 # The size of each row is the size of filter 5*5*3
      ydimen: 1  
      block_decay_mult: 1.5 # the same with breadth_decay_mult
    }
  }
  ```
  - `connectivity_mode` in `LayerParameter` can permanently prune zero-weighted connections: if you want to freeze the zero weights in the weight matrix, please use [connectivity_mode](/src/caffe/proto/caffe.proto#L375).
  - local [regularization_type](/src/caffe/proto/caffe.proto#L316) ("L1/L2") is supported for each `ParamSpec` (e.g. weights) in each layer.
  
During training, you will see some sparsity statistics. The sparsity is shown in the order of layers, and in each layer, in the order of weights and then biases. Basically, it plots sparsity for all parameter blobs in caffe, like parameters for a batch normalization layer. We usually care only about the sparsity of weights.


### Test/Evaluate sparsity-structured deep neural networks 
Testing the SSL learned DNN is also simple by configuring prototxt (i.e. `conv_mode`). Note that ATLAS and OpenBLAS do not officially support sparse BLAS, please use mkl BLAS if you want to use the Compressed Sparse Row feature in CPU mode.
  - [conv_mode](/src/caffe/proto/caffe.proto#L655) in `ConvolutionParameter` configures the implementations of convolution (GEMM, CSR or Concatenation+GEMM). Following is an example to configure `deploy.prototxt` so that the matrix multiplication is operated by (sparse weight matrix) * (dense feature map matrix) (`LOWERED_CSRMM`), GEMM (`LOWERED_GEMM`) or Concatenation+GEMM (`LOWERED_CCNMM`): 
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
    engine: CAFFE # Those features are available in CAFFE cuBLAS mode instead of CUDNN
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


Training only supports `LOWERED_GEMM` mode, which is the default one, but both `CAFFE` and `CUDNN` `engine` are supported for training. 

Note that our code uses standard `caffemodel` to read and store weights, but the weight matrixes of convolutional and fully-connected layers are also snapshotted as `$CAFFE_ROOT/layername.weight` for visualization when you run DNN testing. The `.weight` format obeys [Matrix Market](http://math.nist.gov/MatrixMarket/) format. You can use interfaces of [C](http://math.nist.gov/MatrixMarket/mmio-c.html), Fortran and [Matlab](http://math.nist.gov/MatrixMarket/mmio/matlab/mmiomatlab.html) to read those weight matrixes.

## Tricks
For training large-scale DNNs, the following setups may be a good starting point (which are verified by AlexNet in ImageNet):
  1. Set the base learning rates of both SSL and fine-tuning to `0.1x` of the base learning rate of training original DNNs from stratch.
  2. Set the maximum iteration `K` of SSL to about half of the max iteration `M` of training original DNNs from stratch (`K=M/2`); set max iteration `N` of finetuning to around `M/3`.
  3. During SSL, training with the first learning rate is critical to get high sparsity, please train it longer with, say, `0.7*K`iterations. The group sparsity increase slowly at the early iterations and will ramp up rapidly in the later iterations. In SSL, the training stage under the second learning rate is the key stage that can recover accuracy.
  

## Notes
1. Stabilizing sparsity
  - Note that weights smaller than a threshold ([0.0001](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Sparse_Convolutional_Neural_2015_CVPR_paper.pdf)) are zeroed out after updating weights
2. Caffe version
  - scnn branch is forked from caffe @ commit [eb4ba30](https://github.com/BVLC/caffe/tree/eb4ba30e3c4899edc7a9713158d61503fa8ecf90)
3. cudnn: version of cudnn 5 is supported



## Issues
1. To profile, use `deploy.prototxt` in Python and use `train_val.prototxt` in `caffe time ...`, otherwise, the there might be some bugs in original Caffe. Note that training using `LOWERED_CSRMM` or `LOWERED_CCNMM` is forbidden. `caffe time` calls backward function of each layer, to use `caffe time` to profile, comment backward related codes in `tools/caffe.cpp:time()` function.
2. Speed is compared by matrix-matrix multiplication (GEMM) in each convolutional layer (by MKL BLAS in CPU mode and cuBLAS (not cuDNN) in GPU mode), in a layer-by-layer fashion. The speedup of the total time may be different, because
    - The implementation of lowering convolution to GEMM is not efficient in Caffe, especially in CPU mode.
    - After the time of GEMM is squeezed, the computation time of other layers (e.g. pooling layers) comes to the surface.
    - However, the lowering and pooling can also be optimized. Please refer to [intel branch](https://github.com/wenwei202/caffe/tree/intel).
3. In GPU mode, the lowering operation to shrink feature matrix in `LOWERED_CCNMM` mode is temporally implemented with CPU subroutines. Please [pull request](https://github.com/wenwei202/caffe/issues/3) if you implemented it in GPU mode.
4. `make runtest`: see reports [here](https://github.com/BVLC/caffe/issues/4328#issuecomment-229263764) and related [issue](https://github.com/wenwei202/caffe/issues/2)
5. More in [Caffe Issue 4328](https://github.com/BVLC/caffe/issues/4328)

## Citations

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
