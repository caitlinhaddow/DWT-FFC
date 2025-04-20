
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

WORKDIR /DWT-FFC

# required for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y  

RUN conda install -c conda-forge timm -y
RUN pip install opencv-python scikit-image tensorboardx yacs "numpy<2" kornia pytorch-lightning

COPY . .

ENTRYPOINT ["python"]
# ENTRYPOINT ["bash", "-c"]

CMD ["python", "test.py", "--test_dir", "'/DWT-FFC/input_data/'", "--datasets", "Dense", "--weights", "validation_best.pkl"]

# TO BUILD IMAGE
# hare build -t ceh94/dwtffc .

# --- TO TEST IMAGES ---
hare run --rm --gpus all --shm-size=128g \
--mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/weights,target=/DWT-FFC/weights \
--mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/output_result,target=/DWT-FFC/output_result \
--mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/input_data,target=/DWT-FFC/input_data \
ceh94/dwtffc \
test.py --test_dir /DWT-FFC/input_data/ --datasets "Dense_Haze" "DNH_1600x1200_test" "SMOKE_1600x1200_test" --weights "2025-04-14_23-55-41__NH_NH2_RBm10_epoch01000.pkl" "2025-04-14_23-55-41__NH_NH2_RBm10_epoch02000.pkl" "2025-04-14_23-55-41__NH_NH2_RBm10_epoch03000.pkl" "2025-04-14_23-55-41__NH_NH2_RBm10_epoch04000.pkl" "2025-04-14_23-55-41__NH_NH2_RBm10_epoch05000.pkl" "2025-04-14_23-55-41__NH_NH2_RBm10_epoch06000.pkl" "2025-04-14_23-55-41__NH_NH2_RBm10_epoch07000.pkl" "2025-04-14_23-55-41__NH_NH2_RBm10_epoch08000.pkl" "2025-04-14_23-55-41__NH_NH2_RBm10_epoch09000.pkl" "2025-04-14_23-55-41__NH_NH2_RBm10_epoch10000.pkl"

# --- TO TRAIN MODEL ---
hare run --rm --gpus '"device=2,3"' --shm-size=128g \
--mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/weights,target=/DWT-FFC/weights \
--mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/input_training_data,target=/DWT-FFC/input_training_data \
--mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/check_points,target=/DWT-FFC/check_points \
ceh94/dwtffc \
train.py --datasets _NH_NH2_RBm10

# hare run --rm --gpus all --shm-size=128g \
# --mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/weights,target=/DWT-FFC/weights \
# --mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/input_training_data,target=/DWT-FFC/input_training_data \
# --mount type=bind,source=/mnt/faster0/ceh94/DWT-FFC/check_points,target=/DWT-FFC/check_points \
# ceh94/dwtffc \
# train.py

# --- TO RUN IN INTERACTIVE MODE ---
# hare run --gpus all -it ceh94/dct
# hare run --gpus '"device=0,7"' -it ceh94/dct

# docker system prune ## run every so often to clear out system
