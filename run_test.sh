#!/bin/bash

## CH Dissertation: New File added
## Runs code needed to test the model. Assumes code is on Hex in faster0/ceh94/DWT-FFC - if not needs replacing.
## Datasets and weight files to batch test should be listed after the appropriate parameter as strings separated by spaces only.
## BEFORE RUNNING: give execution permissions with chmod +x run_train.sh
## Then run with ./run_train.sh

set -e  # Exit on error
# set -x  # Echo commands for debugging

# BUILD image
hare build -t ceh94/dwtffc .

# START THE CONTAINER and RUN TEST CODE
hare run --rm --gpus '"device=0,1"' --shm-size=128g \
  --mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/weights,target=/DWT-FFC/weights \
  --mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/output_result,target=/DWT-FFC/output_result \
  --mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/input_data,target=/DWT-FFC/input_data \
  ceh94/dwtffc \
  test.py \
  --output_dir output_result \
  --datasets "Dense_Haze" "DNH_1600x1200_test" "SMOKE_1600x1200_test" "D-Fire" "FIRE" \
  --weights "2025-05-09_17-27-51_DNHDense_epoch01000.pkl" 