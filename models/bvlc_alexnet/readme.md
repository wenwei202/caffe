---
name: BVLC AlexNet Model
caffemodel: bvlc_alexnet.caffemodel
caffemodel_url: http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
license: unrestricted
sha1: 9116a64c0fbe4459d18f4bb6b56d647b63920377
caffe_commit: 709dc15af4a06bebda027c1eb2b3f3e3375d5077
---

## finetune AlexNet with feature sparsity regularization
```
./models/bvlc_alexnet/train_script.sh 0.001 0.0000000009 1 models/bvlc_alexnet/template_solver.prototxt models/bvlc_alexnet/template_train_val.prototxt models/bvlc_alexnet/bvlc_alexnet.caffemodel
```

This model is a replication of the model described in the [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) publication.

Differences:
- not training with the relighting data-augmentation;
- initializing non-zero biases to 0.1 instead of 1 (found necessary for training, as initialization to 1 gave flat loss).

The bundled model is the iteration 360,000 snapshot.
The best validation performance during training was iteration 358,000 with validation accuracy 57.258% and loss 1.83948.
This model obtains a top-1 accuracy 57.1% and a top-5 accuracy 80.2% on the validation set, using just the center crop.
(Using the average of 10 crops, (4 + 1 center) * 2 mirror, should obtain a bit higher accuracy.)

This model was trained by Evan Shelhamer @shelhamer

## License

This model is released for unrestricted use.

## scripts
```
./models/bvlc_alexnet/train_script_force.sh 0.001 0.00005 Degradation 1 models/bvlc_alexnet/template_solver_force.prototxt models/bvlc_alexnet/alexnet_0.57982_split.caffemodel.h5
```

## Email notification when the training is one
You can put your email address in the end of the script (e.g. `train_script_force.sh`),so that you will be notified by an email when the training is done.
Before you can be notified, please install `mail` by
```
sudo apt-get install mailutils
sudo apt-get install sendmail
```
and test it by
```
echo "Test content" | mail -s "Hello" youraddress@example.com
```
The email may be blocked if you send it too frequently.
