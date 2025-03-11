
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

WORKDIR /DWT-FFC

# required for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y  

RUN conda install -c conda-forge timm -y
RUN pip install opencv-python scikit-image tensorboardx yacs "numpy<2" kornia pytorch-lightning

COPY . .

CMD ["bash", "-c", "python test.py --test_dir '/DWT-FFC/input_data/'"]
# CMD ["python", "test.py", "--test_dir", "/DW-GAN/input_data/"] ##try this instead

# TO BUILD IMAGE
# hare build -t ceh94/dwtffc_test .

## TO RUN IMAGE
hare run --rm --gpus '"device=0"' --shm-size=128g \
--mount type=bind,source=/homes/ceh94/DWT-FFC/weights,target=/DWT-FFC/weights \
--mount type=bind,source=/homes/ceh94/DWT-FFC/output_result,target=/DWT-FFC/output_result \
--mount type=bind,source=/homes/ceh94/DWT-FFC/input_data,target=/DWT-FFC/input_data \
ceh94/dwtffc_test

# docker system prune ## run every so often to clear out system