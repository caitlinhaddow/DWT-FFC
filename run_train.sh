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

# START THE CONTAINER and RUN TRAINING CODE
hare run --rm --gpus '"device=2,3"' --shm-size=128g \
  --mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/weights,target=/DWT-FFC/weights \
  --mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/output_result,target=/DWT-FFC/output_result \
  --mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/input_training_data,target=/DWT-FFC/input_training_data \
  --mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/check_points,target=/DWT-FFC/check_points \
  ceh94/dwtffc \
  train.py \
  -train_batch_size 8 \
  --model_save_dir check_points \
  -train_epoch 8005 \
  --datasets "NHNH2RBm10"\



