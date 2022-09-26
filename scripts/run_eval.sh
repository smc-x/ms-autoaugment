#!/bin/bash
export DEVICE_ID=0
export DEVICE_NUM=1
export RANK_ID=0
export RANK_SIZE=1

if [ $# == 2 ]; then
    python ../test.py \
        --checkpoint_path $1 \
        --dataset cifar10 \
        --dataset_path $2 \
        > eval.log 2>&1 &
else
    echo "Usage: \
bash run_eval.sh [CHECKPOINT_PATH] [DATASET_PATH]"
    exit 1
fi
