#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet var data dir

DBPATH=/home/public/imagenet/
DATA=data/ilsvrc12/
TOOLS=build/tools/

TRAIN_DATA_ROOT=/home/public/imagenet/ILSVRC2012_img_train/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

echo "Creating split train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/50000_split_train.txt \
    $DBPATH/ilsvrc12_split_rain_lmdb

echo "Done."
