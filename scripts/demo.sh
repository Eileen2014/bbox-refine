#!/bin/sh
cd ../models

# Set the following variables
GPU_ID=-1
DETECTOR='yolov2'
PROJECT_ROOT=/home/$USER/kv/bbox_refine
SUPER_TYPE=ers_400

CUDA_VISIBLE_DEVICES=${GPU_ID} python baseline.py \
    --project_root ${PROJECT_ROOT} \
    --gpu_id ${GPU_ID} \
    --super_type ${SUPER_TYPE} \
    --detector ${DETECTOR}

