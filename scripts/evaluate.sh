#!/bin/sh
cd ../models

GPU_ID=-1
PROJECT_ROOT=/home/$USER/kv/bbox_refine
SPLIT='val'

# Set the following variables (this decides logs_root)
DETECTOR='yolov2'
SUPER_TYPE=ers_400

# For VOC dataset
DATASET='voc'
DATA_DIR='datasets/VOC2012'

# For YTO dataset
DATASET='yto'
DATA_DIR='datasets/YTOdevkit/YTO'

CUDA_VISIBLE_DEVICES=${GPU_ID} python baseline.py \
    --evaluate \
    --project_root ${PROJECT_ROOT} \
    --gpu_id ${GPU_ID} \
    --split ${SPLIT} \
    --super_type ${SUPER_TYPE} \
    --detector ${DETECTOR} \
    --dataset ${DATASET} \
    --data_dir ${DATA_dIR}