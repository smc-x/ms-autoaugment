#!/bin/bash
rm -rf logs
mkdir ./logs

mpirun -n 8 --allow-run-as-root python train.py \
        --dataset svhn \
        --dataset_path $1 \
        --run_distribute=True \
        --device_target=GPU \
        --lr_init=0.08 \
        --epoch_size=50 \
    > ./logs/distributed_train.log 2>&1 &
