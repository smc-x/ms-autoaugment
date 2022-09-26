#!/bin/bash
if [ $# == 1 ]; then
    python train.py \
        --dataset svhn \
        --dataset_path $1 \
        --device_target=GPU \
        --lr_init=0.01 \
        --epoch_size=50 \
        > train.log 2>&1 &
else
    echo "Usage: bash run_standalone_train_gpu.sh [DATASET_PATH]"
    exit 1
fi
