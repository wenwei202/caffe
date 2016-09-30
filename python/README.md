# Python tools
## resnet_generator.py - generates resnets on cifar-10

e.g.
```
# generate resnets with 3 groups. Refer to section 4.2 in https://arxiv.org/abs/1512.03385
python python/resnet_generator.py \
--net_template examples/cifar10/resnet_template.prototxt \
--n 3

python python/resnet_generator.py \
--net_template examples/cifar10/resnet_template.prototxt \
--n 3 \
--sparsify # add SymmetricRectifyLayer and SparsifyLayer before every ReLULayer

python python/resnet_generator.py \
--net_template examples/cifar10/resnet_template.prototxt \
--n 3 \
--regularize # add SparsifyLayer before ConvolutionLayers except the shortcuts
```

## nn_decomposer.py - decompose convolutional layers to low rank space
e.g.
```
python python/nn_decomposer.py \
--prototxt examples/cifar10/cifar10_full_train_test.prototxt \ # the original network structure
--caffemodel examples/cifar10/cifar10_full_iter_240000_0.8201.caffemodel.h5 \ # the trained caffemodel
--ranks 13,21,27 # the reserved rank in each conv layers
```
Each conv layer will be decompsed to one conv layers (with low-rank basis as the filters) and one 1x1 conv layer (which linearly combines the feature map basis to generate output feature maps with the same dimensionality).
In this example, the network prototxt is saved as `examples/cifar10/cifar10_full_train_test.prototxt.lowrank.prototxt` and corresponding decomposed weights are saved in `examples/cifar10/cifar10_full_iter_240000_0.8201.caffemodel.lowrank.caffemodel`. Note that the original biases are moved to linear combination layer.

More specifically, the conv layer of `conv1`
```
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 32
    pad: 2
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.0001
    }
    bias_filler {
      type: "constant"
    }
  }
}
```
is decomposed to `conv1_lowrank` and `conv1_linear`
```
layer {
  name: "conv1_lowrank"
  type: "Convolution"
  bottom: "data"
  top: "conv1_lowrank"
  param {
    lr_mult: 1.0
  }
  convolution_param {
    num_output: 13
    bias_term: false
    pad: 2
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.0001
    }
  }
}
layer {
  name: "conv1_linear"
  type: "Convolution"
  bottom: "conv1_lowrank"
  top: "conv1"
  param {
    lr_mult: 1.0
  }
  param {
    lr_mult: 2.0
  }
  convolution_param {
    num_output: 32
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.0001
    }
    bias_filler {
      type: "constant"
    }
  }
}
```
## caffemodel_convertor.py - convert the format of caffemodel 
Convert the format of model between `.caffemodel` and `.caffemodel.h5`

e.g.
```
python python/caffemodel_convertor.py \
--network examples/mnist/lenet_train_test.prototxt \
--caffemodel examples/mnist/lenet_0.9917.caffemodel.h5
```

## netsolver.py - python solver 
Similar to `caffe train`, e.g.
```
python python/netsolver.py \
--solver models/bvlc_alexnet/solver.prototxt \
--weights models/bvlc_alexnet/alexnet_0.57982.caffemodel.h5
```
The caffemodel must be hdf5 format, use `caffemodel_convertor.py` to convert `.caffemodel` file to `.caffemodel.h5` file.
