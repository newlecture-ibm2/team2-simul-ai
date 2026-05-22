FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY . .

# FastAPI 서버 실행 (8000번 포트 개방)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
