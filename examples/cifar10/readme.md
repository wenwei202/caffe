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
Use the same [net](https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full_train_test.prototxt) in https://github.com/BVLC/caffe while extend the number of training steps.
1. `cifar10_full_baseline_train_test.prototxt`: the network prototxt.
2. `cifar10_full_baseline_multistep_solver.prototxt` is the corresponding solver.

### SSL to learn high row-sparsity and column-sparsity
1. `cifar10_full_train_test_kernel_shape.prototxt`: the network prototxt enabling group lasso regularization on each row/filter (by setting `breadth_decay_mult`) and each column/FilterShape (by setting `kernel_shape_decay_mult`)
2. Because we need to explore the hyperparameter space (of weight decays, learning rate, etc.), we ease the exploration by [train_script.sh](/examples/cifar10/train_script.sh), whose arguments are hyperparameters we have interest in:
```
./examples/cifar10/train_script.sh \
<base_lr> \ # base learning rate
<weight_decay> \ # traditional weight decay coefficient [L2|L1 is specified in template solver prototxt]
<kernel_shape_decay >\ # group Lasso decay coefficient on columns. DEPRECATED in CPU mode (fill 0.0 here) and use block_group_decay instead
<breadth_decay> \ # group Lasso decay coefficient on rows. DEPRECATED in CPU mode (fill 0.0 here) and use block_group_decay instead
<block_group_decay> \ # group Lasso decay coefficient on sub-blocks tiled in the weight matrix
<device_id> \ # GPU device ID, -1 for CPU
<template_solver.prototxt> \ # the template solver prototxt including all other hyper-parameters. The path is relative to examples/cifar10/
[finetuned.caffemodel/.solverstate] # optional, the .caffemodel to be fine-tuned or the .solverstate to recover training process. The path is relative to examples/cifar10/
```

An example to start SSL training:
```
cd $CAFFE_ROOT
./examples/cifar10/train_script.sh 0.001 0.0 0.003 0.003 0.0 0 \
template_group_solver.prototxt \
yourbaseline.caffemodel
```
`template_group_solver.prototxt` is a template solver whose net is `cifar10_full_train_test_kernel_shape.prototxt`. 

The output and snapshot will be stored in folder named `examples/cifar10/<HYPERPARAMETER_LIST_DATE>` (e.g. `examples/cifar10/0.001_0.0_0.003_0.003_0.0_Fri_Aug_26_14-40-34_PDT_2016` ). Optionally, you can configure the `file_prefix` in `train_script.sh` to change the name of snapshotted models.

`train_script.sh` will generate `examples/cifar10/<HYPERPARAMETER_LIST_DATE>/solver.prototxt` based on input arguments, and the log info will be outputed into file `examples/cifar10/<HYPERPARAMETER_LIST_DATE>/train.info`


### Finetune the model regularized by SSL
Similar to SSL training, but use different network prototxt and solver template.

**Step 1.** Write a network prototxt, which can freeze the compact structure learned by SSL, e.g. [cifar10_full_train_test_ft.prototxt](/examples/cifar10/cifar10_full_train_test_ft.prototxt#L41):
```
  connectivity_mode: DISCONNECTED_GRPWISE # disconnect connections that belong to all-zero rows or columns
```
You can also use `connectivity_mode: DISCONNECTED_ELTWISE` to freeze all weights whose values are zeros.

 **Step 2.** Launch `train_script.sh` to start fine-tuning
```
cd $CAFFE_ROOT
./examples/cifar10/train_script.sh 0.0001 0.004 0.0 0.0 0.0 0 \
template_finetune_solver.prototxt \
yourSSL.caffemodel
```

## ResNets
The process is similar. 
### Some tools
**Tool 1.** ResNets generator - a python tool to generate prototxt for ResNets. Please find it [in our repo](/examples/resnet_generator.py).
```
cd $CAFFE_ROOT
# --n: number of groups, please refer to the https://arxiv.org/abs/1512.03385
# --net_template: network template specifying the data layer
# --connectivity_mode: 0 - CONNECTED; 1 - DISCONNECTED_ELTWISE; 2 - DISCONNECTED_GRPWISE
# --no-learndepth: DO NOT use SSL to learn the depth of resnets
# --learndepth: DO use SSL to learn the depth of resnets
python examples/resnet_generator.py \
--n 3 \
--net_template examples/cifar10/resnet_template.prototxt \
--connectivity_mode 0 \
--no-learndepth
```
The usage of `connectivity_mode` is explained in [caffe.proto](/src/caffe/proto/caffe.proto#L362).
Generated prototxt is `cifar10_resnet_n3.prototxt`

**Tool 2.** Data augmentation (Padding cifar10 images)

Configure [PAD](/examples/cifar10/create_padded_cifar10.sh#L7) and run `create_padded_cifar10.sh`. Note `create_padded_cifar10.sh` will remove `cifar10_train_lmdb` and `cifar10_train_lmdb`, but you can run `create_cifar10.sh` to generate them again.

### Train, SSL regularize and fine-tune ResNets
** Step 1.** Train ResNets baseline 
```
cd $CAFFE_ROOT
./examples/cifar10/train_script.sh 0.1 0.0001 0.0 0.0 0.0 0 \
template_resnet_solver.prototxt 
```
** Step 2.** Regularize the depth of ResNets baseline 

Create or generate a network prototxt (e.g. `cifar10_resnet_n3_depth.prototxt`), where group lasso regularizations are enforced  on the convolutional layers between each pair of shortcut endpoints, 
```
cd $CAFFE_ROOT
python examples/resnet_generator.py \
--n 3 \
--net_template examples/cifar10/resnet_template.prototxt \
--connectivity_mode 0 \
--learndepth
mv examples/cifar10/cifar10_resnet_n3.prototxt examples/cifar10/cifar10_resnet_n3_depth.prototxt
```
then
```
cd $CAFFE_ROOT
./examples/cifar10/train_script.sh 0.1 0.0001 0.0 0.0 0.007 0 \
template_resnet_depth_solver.prototxt \
yourResNetsBaseline.caffemodel
```
** Step 3.** Finetune depth-regularized ResNets

Create or generate a network prototxt similar to `cifar10_resnet_n3_ft.prototxt` by setting `connectivity_mode: DISCONNECTED_GRPWISE`, 
```
cd $CAFFE_ROOT
python examples/resnet_generator.py \
--n 3 \
--net_template examples/cifar10/resnet_template.prototxt \
--connectivity_mode 2 \
--no-learndepth
mv examples/cifar10/cifar10_resnet_n3.prototxt examples/cifar10/cifar10_resnet_n3_ft.prototxt
```

then
```
cd $CAFFE_ROOT
./examples/cifar10/train_script.sh 0.01 0.0001 0.0 0.0 0.0 0 \
template_resnet_finetune_solver.prototxt \
your-depth-Regularized-ResNets.caffemodel
```

## Notes
1. Please explore the hyperparameters of weight decays (by group lasso regularizations) to make the trade-off between accuracy and sparsity. Examples above are good start points.
2. Ignore the huge "Total regularization terms". This is because the internal parameters of Scale layer of Batch Normalization layer are summed.

