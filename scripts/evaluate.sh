#!/bin/sh
cd ../models

# Set the following variables
GPU_ID=0
DETECTOR='yolov2'
SPLIT='val'
PROJECT_ROOT=/home/$USER/kv/bbox_refine
SUPER_TYPE=ers_400

CUDA_VISIBLE_DEVICES=${GPU_ID} python baseline.py \
    --evaluate \
    --project_root ${PROJECT_ROOT} \
    --gpu_id ${GPU_ID} \
    --split ${SPLIT} \
    --super_type ${SUPER_TYPE} \
    --detector ${DETECTOR}
