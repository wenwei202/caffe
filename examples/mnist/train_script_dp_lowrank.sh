#!/bin/bash
# Function: learn from full-rank network to low-rank network by dynamic pruning with force regularization 
# set 0 to eliminate force regularizaion
set -e
set -x

folder="examples/mnist/"
file_prefix="mnist_dp"
#file_prefix="cifar10_resnet"

if [ "$#" -lt 6 ]; then
	echo "Illegal number of parameters"
	echo "Usage: base_lr rank_ratio force_decay force_type device_id template_solver.prototxt orig_caffemodel lra_type [pruning_iter]"
	exit
fi
base_lr=$1
rank_ratio=$2
force_decay=$3
force_type=$4
solver_mode="GPU"
device_id=0
template_solver=$6
orig_caffemodel=$7
lra_type=$8

current_time=$(date)
current_time=${current_time// /_}
current_time=${current_time//:/-}
snapshot_path=$folder/${base_lr}_${rank_ratio}_${force_decay}_${force_type}_${lra_type}_dp_${current_time}
mkdir $snapshot_path

solverfile=$snapshot_path/solver.prototxt

# generate solver prototxt
cat ${template_solver} > $solverfile
echo "base_lr: $base_lr" >> $solverfile
echo "force_decay: $force_decay" >> $solverfile
echo "force_type: \"$force_type\"" >> $solverfile
echo "snapshot_prefix: \"$snapshot_path/$file_prefix\"" >> $solverfile
if [ "$5" -ne "-1" ]; then
	device_id=$5
	echo "device_id: $device_id" >> $solverfile
else
	solver_mode="CPU"
fi
echo "solver_mode: $solver_mode" >> $solverfile

# start training
if [ "$#" -ge "9" ]; then
	python python/lowrank_netsolver.py --solver ${solverfile} --weights ${orig_caffemodel} --device ${device_id} --ratio ${rank_ratio} --lra_type ${lra_type} --pruning_iter $9 > ${snapshot_path}/train.info 2>&1
else
	python python/lowrank_netsolver.py --solver ${solverfile} --weights ${orig_caffemodel} --device ${device_id} --ratio ${rank_ratio}  --lra_type ${lra_type}  > ${snapshot_path}/train.info 2>&1
fi

cat ${snapshot_path}/train.info | grep loss+ | awk '{print $8 " " $11}' > ${snapshot_path}/loss.info
python python/plot_train_info.py --traininfo ${snapshot_path}/train.info
content="$(hostname) done: ${0##*/} ${@}. Results in ${snapshot_path}"
echo ${content} | mail -s "Training done" youremail@example.com
