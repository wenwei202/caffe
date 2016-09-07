# Example of training caffenet by SSL on fully-connected layers
## Files
1. `train_val_fc.prototxt`: network prototxt with `kernel_shape_decay_mult` and `breadth_decay_mult` added into fully-connected layers (fc6, fc7 and fc8);
2. `template_group_solver.prototxt`: the template of solver;
3. `train_script.sh`: the script to launch training.

## To run
```
cd $CAFFE_ROOT
./models/bvlc_reference_caffenet/train_script.sh 0.001 0.0 0.0005 0.0 0.0 0 \
template_group_solver.prototxt \
caffenet_SSL_0.4259.caffemodel
```

More examples in [examples/cifar10](/examples/cifar10)
