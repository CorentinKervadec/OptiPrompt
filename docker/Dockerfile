FROM nvidia/cuda:11.4.3-base-ubuntu20.04

LABEL name Corentin Kervadec "https://registry.sb.upf.edu"

LABEL authors="Corentin Kervadec"
LABEL version="0.10"
LABEL description="CUDA 11.4.3 Pytorch 1.12 image with \
all dependancies requires to conduct LM prompting experiments"

# /!\ If building the image from  macOs, don't forget to add --platform linux/amd64 

# Update
RUN apt-get update

################### I N S T A L L - L I B #############################
# Python builder
RUN apt install -y build-essential
RUN apt install -y python3-pip

# Pytorch 
RUN pip install --no-cache-dir torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# Python utils
RUN pip install --no-cache-dir tqdm termcolor colorama overrides cached-path gdown 
# Scientific libs
RUN pip install --no-cache-dir Cython spacy scipy pandas
# Viz
RUN pip install seaborn
# Fairseq
RUN pip install --no-cache-dir fairseq
# Hugging face Tranformer + acceleration lib
RUN pip install --no-cache-dir transformers accelerate bitsandbytes
RUN pip install datasets apache_beam mwparserfromhell
RUN ln -s /usr/bin/python3 /usr/bin/python
#######################################################################


# Cleaning
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN mkdir /offload_folder