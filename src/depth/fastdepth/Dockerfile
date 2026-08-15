FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pillow numpy tqdm h5py

RUN git clone https://github.com/Hagaik92/FastDepth.git /opt/fastdepth

COPY infer.py /app/infer.py
WORKDIR /app
ENTRYPOINT ["python3", "infer.py"]