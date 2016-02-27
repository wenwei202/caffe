#!/bin/bash
folder="examples/cifar10"

#solverfile="cifar10_full_grouplasso_solver.prototxt"
solverfile="cifar10_full_multistep_solver.prototxt"

#tunedmodel="cifar10_full_step_iter_200000.caffemodel"

current_time=$(date)
#./build/tools/caffe.bin train --solver=$folder/$solverfile --weights=$folder/$tunedmodel --gpu=0  > "${folder}/${current_time}-train.info" 2>&1
./build/tools/caffe.bin train --solver=$folder/$solverfile  --gpu=0  > "${folder}/${current_time}-train.info" 2>&1
cd $folder
finalfiles=$(ls -ltr *caffemodel *.solverstate | awk '{print $9}' | tail -n 2 )
for file in $finalfiles; do
	cp $file "$current_time-$file"
done
