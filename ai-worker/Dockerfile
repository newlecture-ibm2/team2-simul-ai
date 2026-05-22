# =================================================================
# VTO AI Worker Dockerfile (ARM64 / NVIDIA GB10 DGX Spark)
# =================================================================
# CUDA 13.0 기반 ARM64 이미지 사용
FROM nvcr.io/nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    python3.12-dev \
    build-essential \
    gfortran \
    libgl1 \
    libglib2.0-0 \
    git \
    git-lfs \
    wget \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# python 명령어를 python3.12으로 연결
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

WORKDIR /app

# 1단계: Python 패키지 설치 (캐시 레이어 활용)
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir --break-system-packages -r requirements.txt

# 2단계: IDM-VTON 소스코드 클론 (src/, ip_adapter/ 등)
RUN git clone --depth 1 https://github.com/yisol/IDM-VTON.git /app/idm-vton

# 3단계: SAM 3 설치
RUN git clone --depth 1 https://github.com/facebookresearch/sam3.git /app/sam3-repo && \
    cd /app/sam3-repo && \
    pip install --no-cache-dir --break-system-packages -e .

# 4단계: 모델 가중치 사전 다운로드 스크립트 복사
COPY download_models.py .
COPY test_pipeline.py .

# 5단계: 워커 코드 복사
COPY tasks.py .

# Celery Worker 실행 (GPU 연산이므로 pool=solo, concurrency=1)
CMD ["celery", "-A", "tasks", "worker", "--loglevel=info", "-c", "1", "--pool=solo"]
