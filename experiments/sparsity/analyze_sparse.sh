#!/bin/bash
layernames=$(cat $1 | grep 'base_conv_layer.cpp:397' | awk '{print $5 "_" $7 " " $12}' | awk '{print $1}' | sort | uniq)
echo $layernames
for layer in $layernames; do
	totaltime=$(cat  $1 | grep 'base_conv_layer.cpp:397' | awk '{print $5 "_" $7 " " $12}' | grep $layer | awk '{print $2}' | awk '{ sum+=$1} END {print sum}')
	count=$( cat  $1 | grep 'base_conv_layer.cpp:397' | awk '{print $5 "_" $7 " " $12}' | grep $layer | wc -l)
	time_per_iter=$(expr $totaltime / $count )
	echo $layer  $totaltime/$count = $(expr $time_per_iter \* $2) us
done

