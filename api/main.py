from fastapi import FastAPI
from celery import Celery
from pydantic import BaseModel

app = FastAPI(title="VTO AI API Server")

# ai-worker와 동일한 내부 통신망(redis-queue)을 바라보도록 설정
celery_app = Celery('vto_tasks', broker='redis://redis-queue:6379/0', backend='redis://redis-queue:6379/0')

class VTORequest(BaseModel):
    human_image_url: str
    garment_image_url: str

@app.post("/api/v1/try-on")
async def request_vto(req: VTORequest):
    # 1. Celery 큐에 작업 던지기 (비동기)
    # ⚠️ ai-worker의 tasks.py에 정의될 함수 이름('tasks.process_vto')과 정확히 일치해야 합니다.
    task = celery_app.send_task('tasks.process_vto', args=[task.id, req.human_image_url, req.garment_image_url])
    
    # 2. HTTP 요청에 대해서는 0.1초 만에 즉시 응답 (작업 번호 발급)
    return {"status": "accepted", "job_id": task.id}

@app.get("/api/v1/status/{job_id}")
async def get_status(job_id: str):
    # Spring Boot 백엔드에서 3초마다 이 API를 찔러서 진행 상황을 확인합니다.
    res = celery_app.AsyncResult(job_id)
    return {
        "job_id": job_id, 
        "status": res.state, # PENDING, STARTED, SUCCESS, FAILURE 등
        "result": res.result # 완료 시 S3 이미지 URL 등이 담김
    }
