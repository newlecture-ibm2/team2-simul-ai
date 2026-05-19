# 👗 VTO AI 워커 파이프라인 개발 명세서 및 블루프린트

본 문서는 DGX 서버(128GB VRAM) 환경에서 FastAPI와 Celery를 활용하여 비동기 VTO(Virtual Try-On) 작업을 처리하기 위한 AI 워커 파이프라인의 설계 및 구현 명세서입니다.

## 🏗️ 시스템 아키텍처 및 제약사항

1. **인프라 환경:**
   - 128GB VRAM DGX 서버 1대
   - Docker Compose를 통해 2개의 Celery Worker Replica 구성
2. **통신 흐름:**
   - **요청:** Spring Boot 메인 서버 ➡️ FastAPI (HTTP POST)
   - **대기:** FastAPI ➡️ Celery/Redis 작업 큐 등록 후 즉시 `202 Accepted` 응답
   - **처리:** Celery Worker가 GPU 자원을 활용하여 VTO 파이프라인 실행
   - **응답:** Celery Worker ➡️ Spring Boot Webhook Endpoint (`Multipart/form-data`) 직접 전송
3. **최적화 전략:**
   - **VRAM 최적화 (Cold-start 방지):** 모델(`rembg`, `SAM 3`, `IDM-VTO`, `CodeFormer`)은 Worker 실행 시 전역(Global) 공간에 한 번만 로드하여 유지.
   - **디스크 I/O 최소화:** 중간 결과물 파일 쓰기를 생략하고 `io.BytesIO` 및 메모리 상의 `PIL.Image` 객체로만 처리.
4. **장애 대응 (Error Handling):**
   - 어떠한 예외 상황에서도 Worker 프로세스는 종료되지 않음.
   - 실패 시 Spring Boot 웹훅으로 `status: "FAILED"`와 에러 메시지를 전송.

---

## 🛠️ 환경 설정 (requirements.txt)

AI 파이프라인 실행에 필요한 패키지 목록입니다. (실제 환경에 맞게 버전 조정 필요)

```text
# requirements.txt
fastapi
uvicorn
celery
redis
requests
Pillow
rembg
# AI / Vision Models
torch
torchvision
diffusers
transformers
accelerate
# segment-anything, codeformer 등은 별도의 git 레포지토리나 로컬 패키지로 설치한다고 가정
```

---

## 🚀 메인 API 서버 (api/main.py)

FastAPI를 활용해 Spring Boot 서버로부터 작업 요청을 수신하고 Celery 큐에 등록하는 역할만 수행합니다.

```python
# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.tasks import process_vto

app = FastAPI(title="VTO AI Worker API")

class VTORequest(BaseModel):
    job_id: str
    human_image_url: str
    garment_image_url: str

@app.post("/api/v1/vto", status_code=202)
async def create_vto_job(request: VTORequest):
    try:
        # Celery 작업 비동기 큐잉
        task = process_vto.delay(
            request.job_id, 
            request.human_image_url, 
            request.garment_image_url
        )
        return {"message": "Job enqueued successfully", "task_id": task.id, "job_id": request.job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## ⚙️ VTO AI 워커 파이프라인 (api/tasks.py)

AI 모델들을 전역 변수에 로드하여 VRAM에 상주시키고, 5단계의 VTO 파이프라인을 비동기로 처리합니다. 병렬 처리 및 Webhook 전송, 에러 핸들링 로직이 모두 포함되어 있습니다.

```python
# api/tasks.py
import io
import os
import requests
import traceback
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from celery import Celery

# -------------------------------------------------------------------
# 0. Global Model Pre-loading (VRAM Optimization - No Cold Start)
# -------------------------------------------------------------------
import torch
import rembg
# from diffusers import DiffusionPipeline
# from segment_anything import sam_model_registry, SamPredictor
# from codeformer import CodeFormer

print("Loading AI Models into VRAM globally... This happens only once per worker.")
device = "cuda" if torch.cuda.is_available() else "cpu"

# [모델 전역 로딩]
# 1. rembg
rembg_session = rembg.new_session("u2net")

# 2. SAM 3 (Pseudo 코드)
# sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth").to(device)
# sam_predictor = SamPredictor(sam)

# 3. IDM-VTO (Pseudo 코드)
# idm_vto_pipeline = DiffusionPipeline.from_pretrained("yisol/IDM-VTO", torch_dtype=torch.float16).to(device)

# 4. CodeFormer (Pseudo 코드)
# codeformer = CodeFormer(device=device)

print("All models loaded successfully.")

# Celery Initialization
redis_url = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
celery_app = Celery("vto_worker", broker=redis_url, backend=redis_url)

SPRING_BOOT_WEBHOOK_URL = os.getenv("WEBHOOK_URL", "http://192.168.0.5:8080/api/v1/vto/complete")

# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------
def download_image(url: str) -> Image.Image:
    """URL에서 이미지를 다운로드하여 PIL Image로 메모리에 반환합니다."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    return Image.open(response.raw).convert("RGB")

def send_webhook(job_id: str, status: str, result_image: Image.Image = None, error_message: str = None):
    """Spring Boot 웹훅으로 최종 결과물 또는 실패 상태를 Multipart로 전송합니다."""
    data = {"jobId": job_id, "status": status}
    files = None
    
    if error_message:
        data["errorMessage"] = error_message
        
    if result_image and status == "SUCCESS":
        img_byte_arr = io.BytesIO()
        # 디스크가 아닌 메모리(io.BytesIO)상에서 JPEG로 인코딩
        result_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        files = {"file": ("result.jpg", img_byte_arr, "image/jpeg")}
        
    try:
        if files:
            requests.post(SPRING_BOOT_WEBHOOK_URL, data=data, files=files, timeout=30)
        else:
            requests.post(SPRING_BOOT_WEBHOOK_URL, data=data, timeout=30)
    except Exception as e:
        print(f"[Webhook Error] Failed to send webhook for job {job_id}: {e}")

# -------------------------------------------------------------------
# Pipeline Sub-tasks
# -------------------------------------------------------------------
def process_garment(garment_image: Image.Image) -> Image.Image:
    """1악장: 옷 전처리 (배경 제거)"""
    return rembg.remove(garment_image, session=rembg_session)

def extract_human_mask(human_image: Image.Image) -> Image.Image:
    """2악장: 사람 마스크 추출 (SAM 3)"""
    # sam_predictor.set_image(np.array(human_image))
    # masks, _, _ = sam_predictor.predict(point_coords=..., point_labels=...)
    # 프롬프트 예: "shirt", "clothes"
    mask = Image.new("L", human_image.size, 255) # 더미 코드: 하얗게 칠해진 마스크 반환
    return mask

def inpaint_vto(human_image: Image.Image, garment_image: Image.Image, mask: Image.Image) -> Image.Image:
    """3악장: 메인 인페인팅 합성 (IDM-VTO)"""
    negative_prompt = "bad anatomy, extra limbs, mutated hands, artifacts, poorly drawn, deformed, missing limbs, fused fingers"
    
    # result = idm_vto_pipeline(
    #     image=human_image,
    #     mask_image=mask,
    #     garment_image=garment_image,
    #     negative_prompt=negative_prompt
    # ).images[0]
    
    result = human_image.copy() # 더미 코드
    return result

def restore_face(vto_image: Image.Image) -> Image.Image:
    """4악장: 안면 복원 (CodeFormer 또는 GFPGAN)"""
    # result = codeformer.restore(vto_image)
    result = vto_image.copy() # 더미 코드
    return result

# -------------------------------------------------------------------
# Main Celery Task
# -------------------------------------------------------------------
@celery_app.task(name="process_vto", bind=True)
def process_vto(self, job_id: str, human_image_url: str, garment_image_url: str):
    """5단계 핵심 파이프라인 제어 및 실행 함수"""
    try:
        # 입력 파라미터 바탕으로 이미지 다운로드 (Memory)
        human_image = download_image(human_image_url)
        garment_image = download_image(garment_image_url)
        
        # 1악장 & 2악장: 병렬 처리
        # ThreadPoolExecutor를 사용해 I/O/네트워크 대기 없이 모델 추론 동시 진행 (GIL 영향 고려)
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_garment = executor.submit(process_garment, garment_image)
            future_mask = executor.submit(extract_human_mask, human_image)
            
            processed_garment = future_garment.result()
            human_mask = future_mask.result()
            
        # 3악장: 메인 인페인팅 합성
        vto_result = inpaint_vto(human_image, processed_garment, human_mask)
        
        # 4악장: 안면 복원
        final_image = restore_face(vto_result)
        
        # 5악장: Spring Boot 서버로 Multipart 웹훅 직전송
        send_webhook(job_id=job_id, status="SUCCESS", result_image=final_image)
        
        return {"job_id": job_id, "status": "SUCCESS"}
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"[Worker Error] Job {job_id} failed: {error_msg}")
        
        # 에러 핸들링: Worker는 종료되지 않고 FAILED 상태와 에러 메시지를 서버로 회신
        send_webhook(job_id=job_id, status="FAILED", error_message=str(e))
        return {"job_id": job_id, "status": "FAILED", "error": str(e)}

```
