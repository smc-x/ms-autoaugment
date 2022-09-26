#!/bin/bash
export DEVICE_ID=0
export DEVICE_NUM=1
export RANK_ID=0
export RANK_SIZE=1

if [ $# == 1 ]; then
    python ../train.py \
        --dataset cifar10 \
        --dataset_path $1 \
        > train.log 2>&1 &
else
    echo "Usage: \
bash run_standalone_train.sh [DATASET_PATH]"
    exit 1
fi
