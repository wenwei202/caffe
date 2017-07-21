# ABOUT 

This branch is merged to [master](https://github.com/wenwei202/caffe), and maybe out-of-date.

This is a branch working on low-rank deep neural networks for faster evaluation. Related work is accepted in ICCV 2017.

## Paper

[Coordinating Filters for Faster Deep Neural Networks](https://arxiv.org/abs/1703.09746) in ICCV 2017
```
@InProceedings{WWen_2017_ICCV,
author = {Wen, Wei and Xu, Cong and Wu, Chunpeng and Wang, Yandan and Chen, Yiran and Li, Hai},
title = {Coordinating Filters for Faster Deep Neural Networks},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {December},
year = {2017}
}
```

The source code of NIPS 2016 on structurally-sparse DNNs is in the [scnn](https://github.com/wenwei202/caffe/tree/scnn) branch.

## Tutorial
Tutorials on using [python](/python) for low-rank DNNs.
More details will be updated.

## Caffe version
sfm branch is from caffe @ commit [985493e](https://github.com/BVLC/caffe/tree/985493e9ce3e8b61e06c072a16478e6a74e3aa5a)

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
