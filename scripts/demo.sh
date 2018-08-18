#!/bin/sh
cd ../models

# Set the following variables
GPU_ID=-1
DETECTOR='ssd512'

CUDA_VISIBLE_DEVICES=${GPU_ID} python baseline.py \
    --gpu_id ${GPU_ID} \
    --detector ${DETECTOR}

