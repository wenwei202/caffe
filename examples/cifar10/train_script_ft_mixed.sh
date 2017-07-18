#!/bin/bash
# Function: decompose the network to low-rank format and finetune it
set -e
set -x

folder="examples/cifar10/"
file_prefix="cifar10_full"

if [ "$#" -lt 7 ]; then
	echo "Illegal number of parameters"
	echo "Usage: base_lr rank_config1 rank_config2 device_id orig_net orig_caffemodel template_solver.prototxt"
	exit
fi
base_lr=$1
rank_config1=$2
rank_config2=$3
solver_mode="GPU"
device_id=0
orig_net=$5
orig_caffemodel=$6
template_solver=$7

current_time=$(date)
current_time=${current_time// /_}
current_time=${current_time//:/-}
snapshot_path=$folder/${base_lr}_rank_configs_${current_time}
mkdir $snapshot_path

solverfile=$snapshot_path/solver.prototxt

# generate solver prototxt
cat ${template_solver} > $solverfile
echo "base_lr: $base_lr" >> $solverfile
echo "snapshot_prefix: \"$snapshot_path/$file_prefix\"" >> $solverfile
if [ "$4" -ne "-1" ]; then
	device_id=$4
	echo "device_id: $device_id" >> $solverfile
else
	solver_mode="CPU"
fi
echo "solver_mode: $solver_mode" >> $solverfile

new_json1="${snapshot_path}/config_iclr.json"
new_json2="${snapshot_path}/config_cvpr.json"
cp ${rank_config1} ${new_json1}
cp ${rank_config2} ${new_json2}

python python/mixed_decomposer_tuner.py --solver=$solverfile --model ${orig_net}  --weights=${orig_caffemodel} --rank_config1 ${new_json1} --rank_config2 ${new_json2} --device ${device_id} --path ${snapshot_path} >> "${snapshot_path}/train.info" 2>&1

cat ${snapshot_path}/train.info | grep loss+ | awk '{print $8 " " $11}' > ${snapshot_path}/loss.info
python python/plot_train_info.py --traininfo ${snapshot_path}/train.info
content="$(hostname) done: ${0##*/} ${@}. Results in ${snapshot_path}"
echo ${content} | mail -s "Training done" address@example.com
