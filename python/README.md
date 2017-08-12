# Python tools
Use -h [--help] for help.

## caffemodel_convertor.py - convert the format of caffemodel 
Convert the format of model between `.caffemodel` and `.caffemodel.h5`

e.g.
```
python python/caffemodel_convertor.py \
--network examples/mnist/lenet_train_test.prototxt \
--caffemodel examples/mnist/lenet_0.9917.caffemodel.h5
```
In this branch, please use `hdf5` caffe model file **always**.

## group_spitter.py - split the groups in alexnet to two parallel layers
```
python python/group_spitter.py --original_alexnet models/bvlc_alexnet/train_val.prototxt --split_alexnet models/bvlc_alexnet/train_val_split.prototxt --caffemodel models/bvlc_alexnet/bvlc_alexnet.caffemodel.h5
```

## group_merger.py - merge two parallel layers in split AlexNet to one layer with group=2
```
python python/group_merger.py --original_alexnet models/bvlc_alexnet/train_val.prototxt --split_alexnet models/bvlc_alexnet/train_val_split.prototxt --caffemodel models/bvlc_alexnet/bvlc_alexnet_split.caffemodel.h5
```

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
--force_regularization # add force_mult to conv layers except the first conv and shortcuts

```

## nn_decomposer.py - decompose convolutional layers to low rank space
e.g.
```
python python/nn_decomposer.py \
--prototxt examples/cifar10/cifar10_full_train_test.prototxt \ # the original network structure
--caffemodel examples/cifar10/cifar10_full_iter_240000_0.8201.caffemodel.h5 \ # the trained caffemodel
--ranks 13,21,27 # the reserved rank in each conv layers.
```

`--rankratio 0.95` is also supported to reserve 95% information after low-rank approximation. Note that `--rankratio 1.0` can be used to generate a full-rank equivalent network.

`--rank_config` is also supported to decompose only partial layers, see `--rank_config models/bvlc_alexnet/config.json` as an example.

`--except_layers` is also supported to exclude some layers to be decomposed, see `--except_layers examples/cifar10/resnet_except_shortcuts.json` as an example.

Only one of `--rankratio`, `--rank_config` and `--ranks` can be used.

`--lra_type` can specify the type of low rank approximation, default is `pca`. `svd` and `kmeans` are also supported.

Following is an example to decompose resnet-20 on cifar-10:

```
python python/nn_decomposer.py \
--prototxt examples/cifar10/cifar10_resnet_train_test_n3.prototxt \
--caffemodel examples/cifar10/cifar10_resnet20_64000_0.9118.caffemodel \
--rankratio 0.95 \
--lra_type pca \
--except_layers examples/cifar10/resnet_except_shortcuts.json
```

Each conv layer will be decompsed to one conv layer (with low-rank basis as the filters) and one 1x1 conv layer (which linearly combines the feature map basis to generate output feature maps with the same dimensionality).
In the first example of `cifar10_full`, the network prototxt is saved as `examples/cifar10/cifar10_full_train_test.prototxt.lowrank.prototxt` and corresponding decomposed weights are saved in `examples/cifar10/cifar10_full_iter_240000_0.8201.caffemodel.lowrank.caffemodel`. Note that the original biases are moved to linear combination layer.

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

## netsolver.py - python solver 
Similar to `caffe train`, e.g.
```
python python/netsolver.py \
--solver models/bvlc_alexnet/solver.prototxt \
--weights models/bvlc_alexnet/alexnet_0.57982.caffemodel.h5
```
The caffemodel must be hdf5 format, use `caffemodel_convertor.py` to convert `.caffemodel` file to `.caffemodel.h5` file.

## lowrank_netsolver.py - dynamically train a deep neural network with low rank
We also implemented a rank pruning method, which incrementally prunes a rank during the training when the weight approximation error is very small after reducing a rank. 
Usage:
```
python python/lowrank_netsolver.py \
--solver examples/cifar10/cifar10_full_multistep_solver_lowrank.prototxt \ # The solver of full-rank neural network. The full-rank DNNs can be generated by nn_decomposer.py with --rankratio=0.95
--weights examples/cifar10/cifar10_full_0.8201_fullrank.caffemodel.h5 \ # The full-rank caffemodel which can be generated by nn_decomposer.py
--device 1 \ # specify GPU ID, and -1 for CPU
--ratio 0.99 \ # The portion of information to reserve after each pruning
[ --pruning_iter 160000 ] # Only prune low-rank filters in the first 160000 iterations.
```

## lowrank_approx.py
From [C. Tai, ICLR 2016](https://github.com/chengtaipu/lowrankcnn)

Generate lowrank weights
```
python python/lowrank_approx.py --config models/bvlc_alexnet/config_iclr.json --model models/bvlc_alexnet/train_val.prototxt --weights models/bvlc_alexnet/bvlc_alexnet.caffemodel.h5 --save_model models/bvlc_alexnet/train_val_lowrank_iclr.prototxt --save_weights models/bvlc_alexnet/bvlc_alexnet_lowrank_iclr.caffemodel.h5
```

## mixed_decomposer_tuner.py - decompose net by mixed low-rank approximation
e.g. decompose some (the first two) layers by the method of [C. Tai, ICLR 2016](https://github.com/chengtaipu/lowrankcnn) and other layers by our method

```
python python/mixed_decomposer_tuner.py --solver=models/bvlc_alexnet/finetune_solver.prototxt --model models/bvlc_alexnet/train_val.prototxt --weights=models/bvlc_alexnet/bvlc_alexnet.caffemodel.h5 --rank_config1 models/bvlc_alexnet/config_iclr.json --rank_config2 models/bvlc_alexnet/config_iccv.json --device 0
```
Note that the caffe model must be in `hdf5` format
