#!/bin/bash
folder="examples/mnist"
solverfile="lenet_grouplasso_solver.prototxt"
tunedmodel="lenet_iter_10000_0.9915.caffemodel"
current_time=$(date)
current_time=${current_time// /_}
current_time=${current_time//:/-}
#./build/tools/caffe.bin train --solver=$folder/$solverfile  > "${folder}/${current_time}-train.info" 2>&1

./build/tools/caffe.bin train --solver=$folder/$solverfile --weights=$folder/$tunedmodel --gpu=2  > "${folder}/${current_time}-train.info" 2>&1

#snapshot_file="lenet_grouplasso_iter_6000.solverstate"
#./build/tools/caffe.bin train --solver=$folder/$solverfile --snapshot=$folder/$snapshot_file --gpu=2 > "${folder}/${current_time}-train.info" 2>&1

cd $folder
finalfiles=$(ls -ltr *caffemodel *.solverstate | awk '{print $9}' | tail -n 2 )
for file in $finalfiles; do
	cp $file "$current_time-$file"
done
