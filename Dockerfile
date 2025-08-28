
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

WORKDIR /DWT-FFC

# required for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y  

RUN conda install -c conda-forge timm -y
RUN pip install opencv-python scikit-image tensorboardx yacs "numpy<2" kornia pytorch-lightning

COPY . .

ENTRYPOINT ["python"]
# ENTRYPOINT ["bash", "-c"]

# CMD ["python", "test.py", "--test_dir", "'/DWT-FFC/input_data/'", "--datasets", "Dense", "--weights", "validation_best.pkl"]


