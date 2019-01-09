#!/bin/sh
cd ../models

# Set the following variables
GPU_ID=0
DETECTOR='yolov2'
SPLIT='test'
PROJECT_ROOT=/home/$USER/kv/bbox_refine
SUPER_TYPE=ers_400

DATA_DIR=datasets/VOCdevkit/VOC2007

RESULTS_DIR=ssd300_COCO_110000.pth
DETECTIONS_FILE=/home/$USER/kv/ssd.pytorch/results/${RESULTS_DIR}/overall/detections.pkl
ANNOTATION_FILE=/home/$USER/kv/ssd.pytorch/results/${RESULTS_DIR}/overall/groundtruth.pkl

CUDA_VISIBLE_DEVICES=${GPU_ID} python baseline.py \
    --evaluate_from_file \
    --detections_file ${DETECTIONS_FILE} \
    --annotations_file ${ANNOTATION_FILE} \
    --project_root ${PROJECT_ROOT} \
    --gpu_id ${GPU_ID} \
    --split ${SPLIT} \
    --super_type ${SUPER_TYPE} \
    --detector ${DETECTOR} \
    --data_dir ${DATA_DIR}
