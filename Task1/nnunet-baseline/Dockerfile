FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN groupadd -r algorithm && \
    useradd -m --no-log-init -r -g algorithm algorithm && \
    mkdir -p /opt/algorithm /input /output /output/images/tumor-lesion-segmentation  && \
    chown -R algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm utils.py /opt/algorithm/
COPY --chown=algorithm:algorithm nnUNet_results /opt/algorithm/nnUNet_results

RUN python -m pip install --user -U pip && \
    python -m pip install --user -r requirements.txt && \
    mkdir -p /opt/algorithm/nnUNet_raw && \
    mkdir -p /opt/algorithm/nnUNet_preprocessed && \
    mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs && \
    mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/clicksTs && \
    mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result

ENV nnUNet_raw="/opt/algorithm/nnUNet_raw"
ENV nnUNet_preprocessed="/opt/algorithm/nnUNet_preprocessed"
ENV nnUNet_results="/opt/algorithm/nnUNet_results"

ENTRYPOINT ["python", "-m", "process"]
