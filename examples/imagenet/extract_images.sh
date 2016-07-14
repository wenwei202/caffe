#!/bin/bash
tarfile=$(ls -l | grep .tar | awk '{print $9}' | cut -d'.' -f1)
for file in $tarfile; do
	mkdir $file
	tar -xvf ${file}.tar -C $file
	# rm ${file}.tar
done
