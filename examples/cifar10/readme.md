---
title: CIFAR-10 experiments of SSL
category: example
description: Train and test Caffe on CIFAR-10 data.
include_in_docs: true
priority: 5
---

# Experiments on CIFAR-10
## ConvNets
### Baseline
1. `cifar10_full_train_test.prototxt`: the network configuration, dropout on `ip1` is added.
2. `cifar10_full_multistep_solver.prototxt` is the corresponding solver.

### SSL to learn high row-sparsity and column-sparsity
1. `cifar10_full_train_test_kernel_shape.prototxt`: the network configuration enabling group lasso regularization on each row/kernel (by setting `breadth_decay_mult`) and each column/kernelShape (by setting `kernel_shape_decay_mult`)
2. Because we need to explore the hyperparameter space (of weight decays, learning rate, etc.), we ease the exploration by [train_script.sh](/examples/cifar10/train_script.sh), whose arguments are hyperparameters we have interest in:
```
./examples/cifar10/train_script.sh \
<base_lr> \ # base learning rate
<weight_decay> \ # traditional weight decay coefficient [L2|L1 is specified in template solver prototxt]
<kernel_shape_decay >\ # group decay coefficient on columns. DEPRECATED in CPU mode (fill 0.0 here) and use block_group_decay instead
<breadth_decay> \ # group decay coefficient on rows. DEPRECATED in CPU mode (fill 0.0 here) and use block_group_decay instead
<block_group_decay> \ # group decay coefficient on blocks tiled in the weight matrix
<device_id> \ # GPU device ID, -1 for CPU
<template_solver.prototxt> \ # the template solver prototxt including all other hyper-parameters. The path is relative to examples/cifar10/
[finetuned.caffemodel/.solverstate] # optional, the .caffemodel to be fine-tuned or the .solverstate to recover paused training process. The path is relative to examples/cifar10/
```
The output and snapshot data will be stored in folder named as examples/cifar10/<HYPERPARAMETER_LIST_DATE>. Please also configure the `file_prefix` in `train_script.sh` to name the snapshotted models.

An example to start training:
```
cd $CAFFE_ROOT
./examples/cifar10/train_script.sh 0.001 0.0 0.003 0.003 0.0 0 \
template_group_solver.prototxt \
yourbaseline.caffemodel
```
### Finetuning the model regularized by SSL
Similar to SSL, but use different network prototxt and solver template.

1. Write a network prototxt, which can freeze the compress structure learned by SSL, e.g. [cifar10_full_train_test_ft.prototxt](/examples/cifar10/cifar10_full_train_test_ft.prototxt#L41):
```
  connectivity_mode: DISCONNECTED_GRPWISE # disconnect connections that belong to all-zero rows or columns
```
2. Launch `train_script.sh` to start training
```
cd $CAFFE_ROOT
./examples/cifar10/train_script.sh 0.0001 0.004 0.0 0.0 0.0 0 \
template_finetune_solver.prototxt \
yourSSL.caffemodel
```
