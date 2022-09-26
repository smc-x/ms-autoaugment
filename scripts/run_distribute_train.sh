#!/bin/bash
if [ $# != 2 ]; then
    echo "Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH]"
    exit 1
fi

export RANK_TABLE_FILE=$1
export DEVICE_NUM=8
export RANK_SIZE=8

PID_LIST=()
for ((i=0; i<${RANK_SIZE}; i++)); do
    export DEVICE_ID=${i}
    export RANK_ID=${i}
    echo "Start distributed training for rank ${RANK_ID}, device ${DEVICE_ID}"
    python ../train.py \
        --dataset=cifar10 \
        --dataset_path=$2 \
        --run_distribute=True \
        --lr_max=0.8 \
        > train-${i}.log 2>&1 &
    pid=$!
    PID_LIST+=("${pid}")
done

RUN_BACKGROUND=1
if (( RUN_BACKGROUND == 0 )); then
  echo "Waiting for all processes to exit..."
  for pid in ${PID_LIST[*]}; do
      wait ${pid}
      echo "Process ${pid} exited"
  done
fi
