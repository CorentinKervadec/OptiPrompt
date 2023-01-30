FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

LABEL name Corentin Kervadec "https://registry.sb.upf.edu"

LABEL authors="Corentin Kervadec"
LABEL version="0.01"
LABEL description="CUDA Pytorch 1.13 image with \
all dependancies requires to conduct LM prompting experiments"

COPY requirements.txt ./

RUN apt-get update
RUN apt-get install -y build-essential python-dev
RUN pip install --no-cache-dir Cython==0.29.33 spacy==3.5 tqdm==4.63.0 termcolor==1.1.0 pandas==1.5.3 colorama==0.4.1 scipy==1.10.0 overrides==3.1.0 transformers accelerate cached-path==1.1.2 gdown==4.5.1 fairseq

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN mkdir /offload_folder