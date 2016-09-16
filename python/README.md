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

