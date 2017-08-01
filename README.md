# ABOUT 
## Repo summary
### Lower-rank deep neural networks (ICCV 2017)
Paper: [Coordinating Filters for Faster Deep Neural Networks](https://arxiv.org/abs/1703.09746).

source code is in master branch.

### Structurally Sparse Deep Neural Networks (NIPS 2016)
See the source code in branch [scnn](https://github.com/wenwei202/caffe/tree/scnn)

### Direct sparse convolution and guided pruning (ICLR 2017)
Originally in branch [intel](https://github.com/wenwei202/caffe/tree/intel), but merged to [IntelLabs/SkimCaffe](https://github.com/IntelLabs/SkimCaffe) with major contributions by @jspark1105

### Caffe version
Master branch is from caffe @ commit [eb4ba30](https://github.com/BVLC/caffe/commit/eb4ba30e3c4899edc7a9713158d61503fa8ecf90)

## Lower-rank deep neural networks (ICCV 2017)
Tutorials on using [python](/python) for low-rank DNNs. More details will be updated.
If any problems/bugs/questions, you are welcome to open an issue and we will response asap.

Paper: [Coordinating Filters for Faster Deep Neural Networks](https://arxiv.org/abs/1703.09746).

### Training with Force Regularization for Lower-rank DNNs
It is easy to use the code to train DNNs toward lower-rank DNNs.
Only three additional protobuf configurations are required:

1. `force_decay` in `SolverParameter`
2. `force_type` in `SolverParameter`
3. `force_mult` in `ParamSpec`

See details and implementations in [caffe.proto](/src/caffe/proto/caffe.proto#L190-L193) and [SGDSolver](/src/caffe/solvers/sgd_solver.cpp#L223)

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
