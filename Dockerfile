## CH Dissertation: New File added
## Primary method used for creating the code environment. Dockerfile is run by run_train.py and run_test.py

# Base pytorch image
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# Set working directory
WORKDIR /DWT-FFC

# Install packages required for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y  

# Install core requirements
RUN conda install -c conda-forge timm -y
RUN pip install opencv-python scikit-image tensorboardx yacs "numpy<2" kornia pytorch-lightning

# Copy files into container
COPY . .

# Entry point for run code in run_train.py and run_test.py
ENTRYPOINT ["python"]


