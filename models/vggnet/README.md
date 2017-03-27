To run ssl:

```
cd ${CAFFE_ROOT}
./build/tools/caffe train \
-solver models/vggnet/ssl_solver_faster.prototxt \
-weights ${VGGNET_CAFFE_MODEL} \
-gpu 0
```
