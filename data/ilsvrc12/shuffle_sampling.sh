#!/bin/bash
set -e
set -x
if [ $# -lt 1 ]; then
	echo 'usage: file [line #]'
	exit
fi

file=$1
total_lines=$(wc -l ${file} | awk '{print $1}')
line_num=$(expr ${total_lines} / 10)
if [ $# -ge 2 ]; then
	line_num=$2
fi

#cat $file | awk '{print $2}' | uniq -c
output_file=${line_num}_split_${file}
shuf -n $line_num $file > $output_file
echo 'sample distribution:'
cat $output_file | awk '{print $2}' | sort -V | uniq -c
