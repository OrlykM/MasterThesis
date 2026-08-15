FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake git wget unzip \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

RUN git clone https://github.com/princeton-vl/DPVO.git --recursive /DPVO
WORKDIR /DPVO

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -f environment.yml

RUN wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && \
    unzip -q eigen-3.4.0.zip -d thirdparty

RUN /opt/conda/envs/dpvo/bin/pip install --no-build-isolation .

RUN /opt/conda/envs/dpvo/bin/pip install gdown && \
    /opt/conda/envs/dpvo/bin/python -m gdown "https://drive.google.com/uc?id=1dRqftpImtHbbIPNBIseCv9EvrlHEnjhX" -O models.zip && \
    unzip -o -q models.zip && \
    test -f dpvo.pth  # fail the build loudly if the weights didn't actually land

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]