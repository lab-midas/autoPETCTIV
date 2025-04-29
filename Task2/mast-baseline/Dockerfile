FROM --platform=linux/amd64 pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

# Set environment variables for nnUNet and LongiSeg
ENV nnUNet_raw=/opt/app/data/nnUNet_raw
ENV nnUNet_preprocessed=/opt/app/data/nnUNet_preprocessed
ENV nnUNet_results=/opt/app/data/nnUNet_results
ENV LONGISEG_PATH=/opt/app/LongiSeg
ENV LongiSeg_raw=/opt/app/data/LongiSeg_raw
ENV LongiSeg_preprocessed=/opt/app/data/LongiSeg_preprocessed
ENV LongiSeg_results=/opt/app/data/LongiSeg_results

# Create required directories
RUN mkdir -p /opt/app/data/nnUNet_raw \
    /opt/app/data/nnUNet_preprocessed \
    /opt/app/data/nnUNet_results \
    /opt/app/data/LongiSeg_raw \
    /opt/app/data/LongiSeg_preprocessed \
    /opt/app/data/LongiSeg_results

# Create a non-root user
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# Install git and other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    unzip \
    wget \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/app

# Copy requirements and install dependencies
COPY requirements.txt /opt/app/
RUN python -m pip install \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Clone LongiSeg as root
RUN git clone https://github.com/MIC-DKFZ/LongiSeg.git /opt/app/LongiSeg

# Install LongiSeg globally
RUN cd /opt/app/LongiSeg && python -m pip install -e .

# Copy the code
COPY processors /opt/app/processors
COPY utils /opt/app/utils
COPY config /opt/app/config
COPY main.py /opt/app/
COPY longiseg_extensions /opt/app/longiseg_extensions

# Merge LongiSeg Extensions with the LongiSeg Framework
RUN cp -r /opt/app/longiseg_extensions/longiseg/* /opt/app/LongiSeg/longiseg/

# Download pre-trained model
RUN mkdir -p /opt/app/data/nnUNet_results \
    && cd /opt/app/data \
    && wget https://zenodo.org/records/10777532/files/nnUNet_results.zip?download=1 -O nnUNet_results.zip \
    && unzip nnUNet_results.zip \
    && rm nnUNet_results.zip

# Change ownership to user after all operations
RUN chown -R user:user /opt/app

# Switch to user for runtime
USER user

# Set the entrypoint to run main.py
ENTRYPOINT ["python", "main.py"]
