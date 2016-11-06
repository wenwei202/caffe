#!/bin/bash
# Function: decompose the network to low-rank format and finetune it
set -e
set -x

folder="examples/cifar10/"
file_prefix="resnet"

if [ "$#" -lt 6 ]; then
	echo "Illegal number of parameters"
	echo "Usage: base_lr ranks device_id orig_net orig_caffemodel template_solver.prototxt [lra_type]"
	exit
fi
base_lr=$1
ranks=$2
solver_mode="GPU"
device_id=0
orig_net=$4
orig_caffemodel=$5
template_solver=$6
lra_type="pca"
if [ "$#" -ge 7 ]; then
	lra_type=$7
fi
current_time=$(date)
current_time=${current_time// /_}
current_time=${current_time//:/-}
ranks_name=${ranks//,/_}
snapshot_path=$folder/${base_lr}_ranks_${ranks_name}_${lra_type}_${current_time}
mkdir $snapshot_path

solverfile=$snapshot_path/solver.prototxt

# generate solver prototxt
cat ${template_solver} > $solverfile
echo "base_lr: $base_lr" >> $solverfile
echo "snapshot_prefix: \"$snapshot_path/$file_prefix\"" >> $solverfile
if [ "$3" -ne "-1" ]; then
	device_id=$3
	echo "device_id: $device_id" >> $solverfile
else
	solver_mode="CPU"
fi
echo "solver_mode: $solver_mode" >> $solverfile

# generate net and caffemodel
python python/nn_decomposer.py --prototxt ${orig_net} --caffemodel ${orig_caffemodel} --ranks ${ranks} --lra_type ${lra_type} --except_layers examples/cifar10/resnet_except_shortcuts.json  > "${snapshot_path}/train.info" 2>&1
gen_net=${orig_net}.lowrank.prototxt
gen_caffemodel=${orig_caffemodel}.lowrank.caffemodel.h5
mv ${gen_net} $snapshot_path
mv ${gen_caffemodel} $snapshot_path
new_net=${snapshot_path}/$( basename $gen_net )
new_caffemodel=${snapshot_path}/$( basename $gen_caffemodel )
echo "net: \"$new_net\"" >> $solverfile

./build/tools/caffe.bin train --solver=$solverfile --weights=${new_caffemodel}  >> "${snapshot_path}/train.info" 2>&1

cat ${snapshot_path}/train.info | grep loss+ | awk '{print $8 " " $11}' > ${snapshot_path}/loss.info
python python/plot_train_info.py --traininfo ${snapshot_path}/train.info
content="$(hostname) done: ${0##*/} ${@}. Results in ${snapshot_path}"
echo ${content} | mail -s "Training done" address@example.com
