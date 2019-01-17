#!/bin/sh

INPUT_DIR=/home/avisek/kv/datasets/YTOdevkit/YTO/JP_bb1000
SUPERPIXELS=1000
OUTPUT_DIR=/home/avisek/kv/datasets/YTOdevkit/YTO/superpixels/ers_$SUPERPIXELS
mkdir -p ${OUTPUT_DIR}

cd ../../superpixels-revisited
./bin/ers_cli --input ${INPUT_DIR} --output ${OUTPUT_DIR} --csv --contour --superpixels ${SUPERPIXELS} --process
