# ARM 서버에서도 돌아가는 NVIDIA 기본 이미지 사용
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Python 3.10 설치를 위한 구문 추가
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# python 명령어를 python3.10으로 연결
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 나머지 아래 부분은 기존과 동일하게 둡니다.
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["celery", "-A", "tasks", "worker", "--loglevel=info", "-c", "1", "--pool=solo"]
