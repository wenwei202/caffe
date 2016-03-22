#!/bin/bash
set -e
set -x

folder="models/bvlc_reference_caffenet"
file_prefix="caffenet_train"
model_path="models/bvlc_reference_caffenet"

if [ "$#" -lt 4 ]; then
	echo "Illegal number of parameters"
	exit
fi
base_lr=$1
weight_decay=$2
kernel_shape_decay=$3
breadth_decay=$4
#regularization_type="L2"
#if [ "$#" -ge 7 ]; then
#	regularization_type=$7
#fi
gpu_id=0
if [ "$#" -ge 5 ]; then
	gpu_id=$5
fi

current_time=$(date)
current_time=${current_time// /_}
current_time=${current_time//:/-}

snapshot_path=$folder/${base_lr}_${weight_decay}_${kernel_shape_decay}_${breadth_decay}_${current_time}
mkdir $snapshot_path

#solverfile="cifar10_full_finetune_solver.prototxt"
#solverfile="cifar10_full_grouplasso_solver.prototxt"
#solverfile="cifar10_full_multistep_solver.prototxt"
#solverfile="cifar10_full_template_solver.prototxt"
solverfile=$snapshot_path/solver.prototxt
template_file='template_solver.prototxt'
if [ "$#" -ge 7 ]; then
	template_file=$7
fi
cat $folder/${template_file} > $solverfile
echo "kernel_shape_decay: $kernel_shape_decay" >> $solverfile
echo "breadth_decay: $breadth_decay" >> $solverfile
echo "weight_decay: $weight_decay" >> $solverfile
echo "base_lr: $base_lr" >> $solverfile
echo "snapshot_prefix: \"$snapshot_path/$file_prefix\"" >> $solverfile
#echo "regularization_type: \"$regularization_type\"" >> $solverfile
#cat $solverfile

if [ "$#" -ge 6 ]; then
	#tunedmodel="cifar10_full_iter_300000_0.8177.caffemodel"
	#tunedmodel="cifar10_full_iter_300000_0.8212.caffemodel"
	#tunedmodel="cifar10_full_grouplasso_iter_160000_0.7869_4e-4l1_0_6e-3.caffemodel"
	#tunedmodel='Tue_Mar_15_00-00-58_EDT_2016-cifar10_full_grouplasso_iter_120000.caffemodel'
	#tunedmodel='Tue_Mar_15_00-14-56_EDT_2016-cifar10_full_grouplasso_iter_120000.caffemodel'
	#tunedmodel='Tue_Mar_15_00-03-18_EDT_2016-cifar10_full_grouplasso_iter_120000.caffemodel'
	#tunedmodel='Tue_Mar_15_00-10-44_EDT_2016-cifar10_full_grouplasso_iter_120000.caffemodel'
	#tunedmodel='Tue_Mar_15_00-05-15_EDT_2016-cifar10_full_grouplasso_iter_120000.caffemodel'
	tunedmodel=$6
	file_ext=$(echo ${tunedmodel} | rev | cut -d'.' -f 1 | rev)
	if [ "$file_ext" = "caffemodel" ]; then
	  ./build/tools/caffe.bin train --solver=$solverfile --weights=$model_path/$tunedmodel --gpu=$gpu_id  > "${snapshot_path}/train.info" 2>&1
	else
	  ./build/tools/caffe.bin train --solver=$solverfile --snapshot=$model_path/$tunedmodel --gpu=$gpu_id > "${snapshot_path}/train.info" 2>&1
	fi
else
	./build/tools/caffe.bin train --solver=$solverfile  --gpu=$gpu_id  > "${snapshot_path}/train.info" 2>&1
fi

cat ${snapshot_path}/train.info | grep loss+ | awk '{print $8 " " $11}' > ${snapshot_path}/loss.info

#cd $folder
#finalfiles=$(ls -ltr *caffemodel *.solverstate | awk '{print $9}' | tail -n 2 )
#for file in $finalfiles; do
#	cp $file "$current_time-$file"
#done
