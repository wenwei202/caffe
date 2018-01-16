# ABOUT 
## Repo summary
### Lower-rank deep neural networks (ICCV 2017)
Paper: [Coordinating Filters for Faster Deep Neural Networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/Wen_Coordinating_Filters_for_ICCV_2017_paper.pdf).

[Poster](/docs/ICCV17-Poster.pdf) is available.

source code is in this master branch.

### Sparse Deep Neural Networks (NIPS 2016)
See the source code in branch [scnn](https://github.com/wenwei202/caffe/tree/scnn)

### (NIPS 2017 Oral) Ternary Gradients to Reduce Communication in Distributed Deep Learning 
A work to accelerate training. [code](https://github.com/wenwei202/terngrad)

### Direct sparse convolution and guided pruning (ICLR 2017)
Originally in branch [intel](https://github.com/wenwei202/caffe/tree/intel), but merged to [IntelLabs/SkimCaffe](https://github.com/IntelLabs/SkimCaffe) with major contributions by @jspark1105

### Caffe version
Master branch is from caffe @ commit [eb4ba30](https://github.com/BVLC/caffe/commit/eb4ba30e3c4899edc7a9713158d61503fa8ecf90)

## Lower-rank deep neural networks (ICCV 2017)
Tutorials on using python to decompose DNNs to low-rank space is [here](/python). 

If any problems/bugs/questions, you are welcome to open an issue and we will response asap.

Details of Force Regularization is in the Paper: [Coordinating Filters for Faster Deep Neural Networks](https://arxiv.org/abs/1703.09746).

### Training with Force Regularization for Lower-rank DNNs
It is easy to use the code to train DNNs toward lower-rank DNNs.
Only three additional protobuf configurations are required:

1. `force_decay` in `SolverParameter`: Specified in solver. The coefficient to make the trade-off between accuracy and ranks. Larger `force_decay`, smaller ranks and usually lower accuracy.
2. `force_type` in `SolverParameter`: Specified in solver. The kind of force to coordinate filters. `Degradation` - The strength of pairwise attractive force decreases as the distance decreases. This is the L2-norm force in the paper; `Constant` - The strength of pairwise attractive force keeps constant regardless of the distance. This is the L1-norm force in the paper.
3. `force_mult` in `ParamSpec`: Specified for the `param` of weights in each layer. The local multiplier of `force_decay` for filters in a specific layer, i.e., `force_mult*force_decay` is the final coefficient for the specific layer. You can set `force_mult: 0.0` to eliminate force regularization in any layer.

See details and implementations in [caffe.proto](/src/caffe/proto/caffe.proto#L190-L193) and [SGDSolver](/src/caffe/solvers/sgd_solver.cpp#L223)

### Examples
An example of training LeNet with L1-norm force regularization:

```
##############################################################\
# The train/test net with local force decay multiplier       
net: "examples/mnist/lenet_train_test_force.prototxt"        
##############################################################/

test_iter: 100
test_interval: 500
# The base learning rate. For large-scale DNNs, you might try 0.1x smaller base_lr of training the original DNNs from scratch.
base_lr: 0.01
momentum: 0.9
weight_decay: 0.0005

##############################################################\
# The coefficient of force regularization.                   
# The hyper-parameter to tune to make trade-off              
force_decay: 0.001                                           
# The type of force - L1-norm force                          
force_type: "Constant"                                       
##############################################################/

# The learning rate policy
lr_policy: "multistep"
gamma: 0.9
stepvalue: 5000
stepvalue: 7000
stepvalue: 8000
stepvalue: 9000
stepvalue: 9500
# Display every 100 iterations
display: 100
# The maximum number of iterations
max_iter: 10000
# snapshot intermediate results
snapshot: 5000
snapshot_prefix: "examples/mnist/lower_rank_lenet"
snapshot_format: HDF5
solver_mode: GPU
```

Retraining a trained DNN with force regularization might get better results, comparing with training from scratch.

### Hyperparameter
We included the hyperparameter of "lambda_s" for AlexNet in [Figure 6](https://arxiv.org/pdf/1703.09746.pdf). 

### Some open research topics
Force Regularization can squeeze/coordinate weight information to much lower rank space, but after low-rank decomposition with the same precision of approximation, it is more challenging to recover the accuracy from the much more lightweight DNNs. 

## License and Citation
Please cite our ICCV and Caffe if it is useful for your research:

    @InProceedings{Wen_2017_ICCV,
	  author={Wen, Wei and Xu, Cong and Wu, Chunpeng and Wang, Yandan and Chen, Yiran and Li, Hai},
      title={Coordinating Filters for Faster Deep Neural Networks},
	  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
	  month = {October},
	  year = {2017}
    }

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
