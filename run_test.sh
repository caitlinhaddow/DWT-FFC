#!/bin/bash

# BEFORE RUNNING: give execution permissions with chmod +x run_train.sh
# Then run with ./run_train.sh

set -e  # Exit on error
# set -x  # Echo commands for debugging

# BUILD image
hare build -t ceh94/dwtffc .

# '"device=0,1"'  'all'

# START THE CONTAINER and RUN TEST CODE
# hare run --rm --gpus '"device=0,1"' --shm-size=128g \
#   --mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/weights,target=/DWT-FFC/weights \
#   --mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/output_result,target=/DWT-FFC/output_result \
#   --mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/input_data,target=/DWT-FFC/input_data \
#   ceh94/dwtffc \
#   test.py \
#   --output_dir output_result \
#   --datasets "NH" "NH2" "SMOKE_1600x1200_test" \
#   --weights "2025-05-09_17-27-51_DNHDense_epoch01000.pkl" 

  ## DATASETS: "NH" "NH2" "SMOKE_1600x1200_test" "Dense_Haze" "DNH_1600x1200_test" "SMOKE_1600x1200_test"

hare run --rm --gpus '"device=0,1"' --shm-size=128g \
  --mount type=bind,source=/mnt/fast0/ceh94/DWT-FFC/weights,target=/DWT-FFC/weights \
  --mount type=bind,source=/mnt/fast0/ceh94/DWT-FFC/output_result,target=/DWT-FFC/output_result \
  --mount type=bind,source=/mnt/fast0/ceh94/DWT-FFC/input_data,target=/DWT-FFC/input_data \
  ceh94/dwtffc \
  test.py \
  --output_dir output_result \
  --datasets "Dense_Haze" "DNH_1600x1200_test" "SMOKE_1600x1200_test" \
  --weights "2025-07-31_23-11-24_NHNH2RBm5_epoch01000.pkl" "2025-07-31_23-11-24_NHNH2RBm5_epoch02000.pkl" "2025-07-31_23-11-24_NHNH2RBm5_epoch03000.pkl" "2025-07-31_23-11-24_NHNH2RBm5_epoch04000.pkl" "2025-07-31_23-11-24_NHNH2RBm5_epoch05000.pkl" "2025-07-31_23-11-24_NHNH2RBm5_epoch06000.pkl" "2025-07-31_23-11-24_NHNH2RBm5_epoch07000.pkl" "2025-07-31_23-11-24_NHNH2RBm5_epoch08000.pkl" "2025-08-02_14-29-25_NHNH2RBm15_epoch01000.pkl" "2025-08-02_14-29-25_NHNH2RBm15_epoch02000.pkl" "2025-08-02_14-29-25_NHNH2RBm15_epoch03000.pkl" "2025-08-02_14-29-25_NHNH2RBm15_epoch04000.pkl" "2025-08-02_14-29-25_NHNH2RBm15_epoch05000.pkl" "2025-08-02_14-29-25_NHNH2RBm15_epoch06000.pkl" "2025-08-02_14-29-25_NHNH2RBm15_epoch07000.pkl" "2025-08-02_14-29-25_NHNH2RBm15_epoch08000.pkl" "2025-08-04_10-39-52_NHNH2RBm20_epoch01000.pkl" "2025-08-04_10-39-52_NHNH2RBm20_epoch02000.pkl" "2025-08-04_10-39-52_NHNH2RBm20_epoch03000.pkl" "2025-08-04_10-39-52_NHNH2RBm20_epoch04000.pkl" "2025-08-04_10-39-52_NHNH2RBm20_epoch05000.pkl" "2025-08-04_10-39-52_NHNH2RBm20_epoch06000.pkl" "2025-08-04_10-39-52_NHNH2RBm20_epoch07000.pkl" "2025-08-04_10-39-52_NHNH2RBm20_epoch08000.pkl"



# TO BUILD IMAGE
# hare build -t ceh94/dwtffc .

# --- TO TEST IMAGES ---
# hare run --rm --gpus all --shm-size=128g \
# --mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/weights,target=/DWT-FFC/weights \
# --mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/output_result,target=/DWT-FFC/output_result \
# --mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/input_data,target=/DWT-FFC/input_data \
# ceh94/dwtffc \
# test.py --test_dir /DWT-FFC/input_data/ --datasets "Dense_Haze" "DNH_1600x1200_test" "SMOKE_1600x1200_test" --weights "2025-04-14_23-55-41__NH_NH2_RBm10_epoch01000.pkl" "2025-04-14_23-55-41__NH_NH2_RBm10_epoch02000.pkl" "2025-04-14_23-55-41__NH_NH2_RBm10_epoch03000.pkl" "2025-04-14_23-55-41__NH_NH2_RBm10_epoch04000.pkl" "2025-04-14_23-55-41__NH_NH2_RBm10_epoch05000.pkl" "2025-04-14_23-55-41__NH_NH2_RBm10_epoch06000.pkl" "2025-04-14_23-55-41__NH_NH2_RBm10_epoch07000.pkl" "2025-04-14_23-55-41__NH_NH2_RBm10_epoch08000.pkl" "2025-04-14_23-55-41__NH_NH2_RBm10_epoch09000.pkl" "2025-04-14_23-55-41__NH_NH2_RBm10_epoch10000.pkl"


## faster download: rsync -uav ceh94@parkin.cs.bath.ac.uk:/mnt/faster0/ceh94/DW-GAN/output_result ~/Documents/Datasets/AA_Results/DWGAN_Results
