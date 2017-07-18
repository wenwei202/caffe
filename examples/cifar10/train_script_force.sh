#!/bin/bash
# train the original network with force regularization
set -e
set -x

folder="examples/cifar10/"
file_prefix="cifar10_full"
#file_prefix="cifar10_resnet"

if [ "$#" -lt 5 ]; then
	echo "Illegal number of parameters"
	echo "Usage: train_script base_lr force_decay force_type  device_id template_solver.prototxt [finetuned.caffemodel/.solverstate]"
	exit
fi
base_lr=$1
force_decay=$2
force_type=$3
solver_mode="GPU"
device_id=0

current_time=$(date)
current_time=${current_time// /_}
current_time=${current_time//:/-}

snapshot_path=$folder/${base_lr}_forcedecay_${force_decay}_${force_type}_${current_time}
mkdir $snapshot_path

solverfile=$snapshot_path/solver.prototxt
template_solver=$5

cat ${template_solver} > $solverfile
echo "base_lr: $base_lr" >> $solverfile
echo "force_decay: $force_decay" >> $solverfile
echo "force_type: \"${force_type}\"" >> $solverfile
echo "snapshot_prefix: \"$snapshot_path/$file_prefix\"" >> $solverfile
if [ "$4" -ne "-1" ]; then
	device_id=$4
	echo "device_id: $device_id" >> $solverfile
else
	solver_mode="CPU"
fi
echo "solver_mode: $solver_mode" >> $solverfile

if [ "$#" -ge 6 ]; then
	tunedmodel=$6
	#file_ext=$(echo ${tunedmodel} | rev | cut -d'.' -f 1 | rev)
	if [[ $tunedmodel == *"caffemodel"* ]]; then
	  ./build/tools/caffe.bin train --solver=$solverfile --weights=$tunedmodel  > "${snapshot_path}/train.info" 2>&1
	else
	  ./build/tools/caffe.bin train --solver=$solverfile --snapshot=$tunedmodel > "${snapshot_path}/train.info" 2>&1
	fi
else
	./build/tools/caffe.bin train --solver=$solverfile   > "${snapshot_path}/train.info" 2>&1
fi

cat ${snapshot_path}/train.info | grep loss+ | awk '{print $8 " " $11}' > ${snapshot_path}/loss.info
python python/plot_train_info.py --traininfo ${snapshot_path}/train.info
