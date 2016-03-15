#!/bin/bash
set -e
set -x

folder="examples/cifar10"
file_prefix="cifar10_full"
model_path="examples/cifar10"

if [ "$#" -lt 3 ]; then
	echo "Illegal number of parameters"
	exit
fi

kernel_shape_decay=$1
breadth_decay=$2
weight_decay=$3

gpu_id=0
if [ "$#" -ge 4 ]; then
	gpu_id=$4
fi

current_time=$(date)
current_time=${current_time// /_}
current_time=${current_time//:/-}

snapshot_path=$folder/$current_time
mkdir $snapshot_path

#solverfile="cifar10_full_finetune_solver.prototxt"
#solverfile="cifar10_full_grouplasso_solver.prototxt"
#solverfile="cifar10_full_multistep_solver.prototxt"
#solverfile="cifar10_full_template_solver.prototxt"
solverfile=$snapshot_path/solver.prototxt
cat $folder/template_solver.prototxt > $solverfile
echo "kernel_shape_decay: $kernel_shape_decay" >> $solverfile
echo "breadth_decay: $breadth_decay" >> $solverfile
echo "weight_decay: $weight_decay" >> $solverfile
echo "snapshot_prefix: \"$snapshot_path/$file_prefix\"" >> $solverfile
#cat $solverfile

if [ "$#" -ge 5 ]; then
	#tunedmodel="cifar10_full_iter_300000_0.8177.caffemodel"
	#tunedmodel="cifar10_full_iter_300000_0.8212.caffemodel"
	#tunedmodel="cifar10_full_grouplasso_iter_160000_0.7869_4e-4l1_0_6e-3.caffemodel"
	#tunedmodel='Tue_Mar_15_00-00-58_EDT_2016-cifar10_full_grouplasso_iter_120000.caffemodel'
	#tunedmodel='Tue_Mar_15_00-14-56_EDT_2016-cifar10_full_grouplasso_iter_120000.caffemodel'
	#tunedmodel='Tue_Mar_15_00-03-18_EDT_2016-cifar10_full_grouplasso_iter_120000.caffemodel'
	#tunedmodel='Tue_Mar_15_00-10-44_EDT_2016-cifar10_full_grouplasso_iter_120000.caffemodel'
	#tunedmodel='Tue_Mar_15_00-05-15_EDT_2016-cifar10_full_grouplasso_iter_120000.caffemodel'
	tunedmodel=$5
	./build/tools/caffe.bin train --solver=$solverfile --weights=$model_path/$tunedmodel --gpu=$gpu_id  > "${snapshot_path}/train.info" 2>&1
	#snapshot_file="cifar10_full_iter_210000.solverstate"
	#./build/tools/caffe.bin train --solver=$solverfile --snapshot=$model_path/$snapshot_file --gpu=$gpu_id > "${snapshot_path}/train.info" 2>&1
else
	./build/tools/caffe.bin train --solver=$solverfile  --gpu=$gpu_id  > "${snapshot_path}/train.info" 2>&1
fi

#cd $folder
#finalfiles=$(ls -ltr *caffemodel *.solverstate | awk '{print $9}' | tail -n 2 )
#for file in $finalfiles; do
#	cp $file "$current_time-$file"
#done
