# DPVO -- built from the official princeton-vl/DPVO source directly (not via
# DPVO_Docker's wrapper scripts, so every fix below is baked into a single
# reproducible Dockerfile instead of a chain of manual docker exec commands).
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake git wget unzip \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

RUN git clone https://github.com/princeton-vl/DPVO.git --recursive /DPVO
WORKDIR /DPVO

# Accept conda Terms of Service -- current conda refuses non-interactive env
# creation from the default channels otherwise (this broke our very first
# DPVO_Docker attempt, this fixes it at the source).
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -f environment.yml

RUN wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && \
    unzip -q eigen-3.4.0.zip -d thirdparty

# --no-build-isolation: pip's default isolated build venv doesn't have torch,
# and DPVO's setup.py imports torch to determine CUDA compile flags. Building
# in the active conda env (which does have torch) fixes this.
#
# TORCH_CUDA_ARCH_LIST must be set explicitly: `docker build` has no GPU
# access (unlike `docker run --gpus`), so PyTorch's auto-detection of which
# architecture to compile for finds zero visible cards and crashes with
# "IndexError: list index out of range" on an empty arch list. 8.9 = Ada
# Lovelace (RTX 40-series, incl. the RTX 4060) -- change this if building
# for a different GPU generation.
ENV TORCH_CUDA_ARCH_LIST="8.9"
RUN /opt/conda/envs/dpvo/bin/pip install --no-build-isolation .

# The official download_models_and_data.sh uses dead Dropbox links (confirmed
# broken -- serves HTML instead of a real zip). Pull the real weights from
# the Google Drive link the README lists as the alternative.
RUN /opt/conda/envs/dpvo/bin/pip install gdown && \
    /opt/conda/envs/dpvo/bin/python -m gdown "https://drive.google.com/uc?id=1dRqftpImtHbbIPNBIseCv9EvrlHEnjhX" -O models.zip && \
    unzip -o -q models.zip && \
    test -f dpvo.pth  # fail the build loudly if the weights didn't actually land

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]