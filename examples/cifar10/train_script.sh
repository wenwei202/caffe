#!/bin/bash
set -e
set -x

folder="examples/cifar10/"
file_prefix="cifar10_your_prefix"
#file_prefix="cifar10_resnet"
#file_prefix="cifar10_full"
model_path="examples/cifar10/"

if [ "$#" -lt 7 ]; then
	echo "Illegal number of parameters"
	echo "Usage: train_script base_lr weight_decay kernel_shape_decay breadth_decay block_group_decay device_id template_solver.prototxt [finetuned.caffemodel/.solverstate]"
	exit
fi
base_lr=$1
weight_decay=$2
kernel_shape_decay=$3
breadth_decay=$4
block_group_decay=$5
solver_mode="GPU"
device_id=0

current_time=$(date)
current_time=${current_time// /_}
current_time=${current_time//:/-}

snapshot_path=$folder/${base_lr}_${weight_decay}_${kernel_shape_decay}_${breadth_decay}_${block_group_decay}_${current_time}
mkdir $snapshot_path

solverfile=$snapshot_path/solver.prototxt
template_file='template_solver.prototxt'
#if [ "$#" -ge 7 ]; then
template_file=$7
#fi

cat $folder/${template_file} > $solverfile
echo "block_group_decay: $block_group_decay" >> $solverfile
echo "kernel_shape_decay: $kernel_shape_decay" >> $solverfile
echo "breadth_decay: $breadth_decay" >> $solverfile
echo "weight_decay: $weight_decay" >> $solverfile
echo "base_lr: $base_lr" >> $solverfile
echo "snapshot_prefix: \"$snapshot_path/$file_prefix\"" >> $solverfile
#if [ "$#" -ge 6 ]; then
if [ "$6" -ne "-1" ]; then
	device_id=$6
	echo "device_id: $device_id" >> $solverfile
else
	solver_mode="CPU"
fi
#fi
echo "solver_mode: $solver_mode" >> $solverfile
#echo "regularization_type: \"$regularization_type\"" >> $solverfile
#cat $solverfile

if [ "$#" -ge 8 ]; then
	tunedmodel=$8
	file_ext=$(echo ${tunedmodel} | rev | cut -d'.' -f 1 | rev)
	if [ "$file_ext" = "caffemodel" ]; then
	  ./build/tools/caffe.bin train --solver=$solverfile --weights=$model_path/$tunedmodel  > "${snapshot_path}/train.info" 2>&1
	else
	  ./build/tools/caffe.bin train --solver=$solverfile --snapshot=$model_path/$tunedmodel > "${snapshot_path}/train.info" 2>&1
	fi
else
	./build/tools/caffe.bin train --solver=$solverfile   > "${snapshot_path}/train.info" 2>&1
fi

cat ${snapshot_path}/train.info | grep loss+ | awk '{print $8 " " $11}' > ${snapshot_path}/loss.info

#cd $folder
#finalfiles=$(ls -ltr *caffemodel *.solverstate | awk '{print $9}' | tail -n 2 )
#for file in $finalfiles; do
#	cp $file "$current_time-$file"
#done
