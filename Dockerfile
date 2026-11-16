FROM huggingface/transformers-pytorch-gpu@sha256:5273b08b13dd4e4931e57842c6955a15c752d95db612c70ccc1dd8e71f534636

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=5000

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY workspace /workspace

EXPOSE 5000

CMD ["python3", "/workspace/app.py"]
