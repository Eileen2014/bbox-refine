#!/bin/sh

INPUT_DIR=/home/avisek/kv/datasets/VOCdevkit/VOC2007/JPEGImages
SUPERPIXELS=1000
OUTPUT_DIR=/home/avisek/kv/datasets/VOCdevkit/VOC2007/superpixels/ers_400
mkdir -p ${OUTPUT_DIR}

cd ../../superpixels-revisited
./bin/ers_cli --input ${INPUT_DIR} --output ${OUTPUT_DIR} --csv --contour --superpixels ${SUPERPIXELS} --process
