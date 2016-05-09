#!/bin/bash
#layernames=$(cat $1 | grep 'base_conv_layer.cpp:414' | awk '{print $5 "_" $7 " " $11}' | awk '{print $1}' | sort | uniq)
#set -x
#set -e
function divfunc () {
    echo "scale = 2; $1 / $2" | bc
}
function multifunc () {
    echo "scale = 2; $1 * $2" | bc
}

if [ $3 = "dense"  ]; then
	pattern='(Dense Scheme Timing)'
else
	if [ $3 = "crs" ]; then
		pattern='(Compressed Row Storage Timing)'
	else
		if [ $3 = "lowering" ]; then
			pattern='(Lowering Over Matrix Multiplying Timing)'
		else
			pattern='(Concatenation Timing)'
		fi
	fi
fi

echo $pattern
#pattern='(Column Concatenation Timing)'

layernames=$(cat $1 | grep "${pattern}" | awk '{print $5 "_" $7 " " $8}' | awk '{print $1}' | sort | uniq)
echo $layernames
for layer in $layernames; do
	totaltime=$(cat  $1 | grep "${pattern}" | awk '{print $5 "_" $7 " " $8}' | grep $layer | awk '{print $2}' | awk '{ sum+=$1} END {printf "%.f",sum}')
	count=$( cat  $1 | grep "${pattern}" | awk '{print $5 "_" $7 " " $8}' | grep $layer | wc -l)
	#time_per_iter=$(expr $totaltime / $count )
	time_per_iter=$(divfunc $totaltime $count )
	if [ $3 != "lowering" ]; then
		#echo $layer  $totaltime/$count = $(expr $time_per_iter \* $2) us
		#echo -e $layer  $totaltime/$count*$2 = '\t' $(multifunc $time_per_iter  $2) '\t' us
		echo -e  $(multifunc $time_per_iter  $2) 
	else
		#echo $layer  $totaltime/$count = $(expr $time_per_iter \* $2) %
		echo $layer  $totaltime/$count*$2 = $(multifunc $time_per_iter  $2) %
	fi
done

