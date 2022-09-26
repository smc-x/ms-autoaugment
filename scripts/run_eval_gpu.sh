#!/bin/bash
if [ $# == 2 ]; then
    python test.py \
        --checkpoint_path $1 \
        --dataset svhn \
        --dataset_path $2 \
        --device_target=GPU \
        > eval.log 2>&1 &
else
    echo "Usage:bash run_eval.sh [CHECKPOINT_PATH] [DATASET_PATH]"
    exit 1
fi
