#!/bin/bash
set -e
set -x

folder="examples/cifar10/"
file_prefix="cifar10_full"

if [ "$#" -lt 5 ]; then
	echo "Illegal number of parameters"
	echo "Usage: train_script base_lr feature_decay device_id template_solver.prototxt template_network.prototxt [finetuned.caffemodel/.solverstate]"
	exit
fi
base_lr=$1
feature_decay=$2
solver_mode="GPU"
device_id=0

current_time=$(date)
current_time=${current_time// /_}
current_time=${current_time//:/-}

snapshot_path=$folder/${base_lr}_featuredecay_${feature_decay}_${current_time}
mkdir $snapshot_path

solverfile=$snapshot_path/solver.prototxt
template_solver=$4
template_network=$5

python python/set_layer_param.py \
--net_template $template_network \
--layer_type Sparsify \
--param_value $feature_decay 
mv ${folder}/generated.prototxt ${snapshot_path}/net.prototxt

cat ${template_solver} > $solverfile
echo "net: \"${snapshot_path}/net.prototxt\"" >> $solverfile
echo "base_lr: $base_lr" >> $solverfile
echo "snapshot_prefix: \"$snapshot_path/$file_prefix\"" >> $solverfile
if [ "$3" -ne "-1" ]; then
	device_id=$3
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
content="$(hostname) done: ${0##*/} ${@}. Results in ${snapshot_path}"
echo ${content} | mail -s "Training done" youremail@example.com
