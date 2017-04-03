To run ssl:

```
cd ${CAFFE_ROOT}
./build/tools/caffe train \
-solver models/vggnet/ssl_solver_faster.prototxt \
-weights ${VGGNET_CAFFE_MODEL} \
-gpu 0,1 # change batch size according to the number of gpus being used.
```
