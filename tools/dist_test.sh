#!/usr/bin/env bash

PYTHON=/home/cver/anaconda3/envs/AerialDetection/bin/python


CONFIG=$1
CHECKPOINT=$2
GPUS=$3

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
