# Dockerfile
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ARG USER_ID=1000
ARG GROUP_ID=1000

# Create non-root user
RUN groupadd -g ${GROUP_ID} appuser && useradd -m -u ${USER_ID} -g ${GROUP_ID} appuser

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DATA_ROOT=/workspace/data \
    OUTPUT_DIR=/workspace/output

# Create common dirs
RUN mkdir -p ${DATA_ROOT} ${OUTPUT_DIR}

# Useful system deps + tini for proper signal handling
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates tini && \
    rm -rf /var/lib/apt/lists/*

# Pre-install Python deps (cache layer)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# We'll mount the repo at /workspace via docker-compose; no COPY of code needed.
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["bash"]   # compose will override with the training command
USER appuser
