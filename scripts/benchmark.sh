#!/bin/sh
cd ../utils

GPU_ID=-1
PROJECT_ROOT=/home/$USER/kv/bbox_refine

# Set the following variables
DETECTOR='yolov2'
SUPER_TYPE=ers_1000

CUDA_VISIBLE_DEVICES=${GPU_ID} python metrics.py\
    --project_root ${PROJECT_ROOT} \
    --gpu_id ${GPU_ID} \
    --super_type ${SUPER_TYPE} \
    --detector ${DETECTOR}
