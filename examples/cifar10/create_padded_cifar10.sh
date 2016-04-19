#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

EXAMPLE=examples/cifar10
DATA=data/cifar10
DBTYPE=lmdb
PAD=4

echo "Creating $DBTYPE..."
echo "removing $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/cifar10_test_$DBTYPE"

rm -rf $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/cifar10_test_$DBTYPE

./build/examples/cifar10/convert_cifar_data.bin $DATA $EXAMPLE $DBTYPE $PAD

mv $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/cifar10_pad${PAD}_train_$DBTYPE
mv $EXAMPLE/cifar10_test_$DBTYPE $EXAMPLE/cifar10_pad${PAD}_test_$DBTYPE 

echo "Computing image mean..."

./build/tools/compute_image_mean -backend=$DBTYPE \
  $EXAMPLE/cifar10_pad${PAD}_train_$DBTYPE $EXAMPLE/pad${PAD}_mean.binaryproto

echo "Done."
