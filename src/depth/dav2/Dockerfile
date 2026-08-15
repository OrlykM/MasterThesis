FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

RUN pip install --no-cache-dir transformers==4.46.3 huggingface_hub pillow numpy tqdm

COPY infer.py /app/infer.py
WORKDIR /app
ENTRYPOINT ["python3", "infer.py"]