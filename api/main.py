from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from celery import Celery

app = FastAPI(title="VTO AI API Server")

# ai-worker와 동일한 내부 통신망(redis-queue)을 바라보도록 설정
celery_app = Celery('vto_worker', broker='redis://redis-queue:6379/0', backend='redis://redis-queue:6379/0')

class VTORequest(BaseModel):
    job_id: str
    human_image_url: str
    garment_image_url: str

@app.post("/api/v1/vto", status_code=202)
async def create_vto_job(request: VTORequest):
    try:
        # 1. Celery 큐에 작업 던지기 (비동기)
        # ai-worker에 등록된 process_vto 호출
        task = celery_app.send_task(
            'process_vto', 
            args=[request.job_id, request.human_image_url, request.garment_image_url]
        )
        return {"message": "Job enqueued successfully", "task_id": task.id, "job_id": request.job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
