"""
VTO AI Worker - Celery Task Pipeline
=====================================
5단계 가상 피팅(Virtual Try-On) 파이프라인을 실행하는 Celery 워커.

모델 로딩 순서:
  0. 전역(Global) 레벨에서 모든 AI 모델을 VRAM에 1회 적재 (Cold-start 방지)
  1악장: rembg - 옷 이미지 배경 제거
  2악장: SAM 3 - 인물 이미지에서 옷 영역 마스크 추출 (텍스트 프롬프트)
  3악장: IDM-VTON - 메인 인페인팅 합성
  4악장: (선택) 안면 복원 - 현재 비활성
  5악장: Spring Boot 웹훅으로 결과 전송
"""
import io
import os
import sys
import requests
import traceback
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List

from PIL import Image
from celery import Celery
import torch
from torchvision import transforms

# -------------------------------------------------------------------
# IDM-VTON 소스 경로 추가
# -------------------------------------------------------------------
IDM_VTON_PATH = "/app/idm-vton"
sys.path.insert(0, IDM_VTON_PATH)

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref

from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL
import rembg

# -------------------------------------------------------------------
# 0. Global Model Pre-loading (VRAM Optimization - No Cold Start)
# -------------------------------------------------------------------
print("=" * 60)
print("[Worker Init] AI 모델을 VRAM에 적재합니다... (최초 1회)")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
weight_dtype = torch.float16

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/app/models")
IDM_VTON_MODEL_PATH = os.path.join(MODEL_CACHE_DIR, "idm-vton")

# ---- [1] rembg 세션 ----
print("[Model Load] rembg (u2net) 로딩...")
rembg_session = rembg.new_session("u2net")

# ---- [2] SAM 3 ----
print("[Model Load] SAM 3 로딩...")
SAM3_CHECKPOINT = os.path.join(MODEL_CACHE_DIR, "sam3", "sam3.pt")

# SAM 3 경로 추가
SAM3_REPO_PATH = "/app/sam3-repo"
sys.path.insert(0, SAM3_REPO_PATH)

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

sam3_model = build_sam3_image_model(checkpoint_path=SAM3_CHECKPOINT)
sam3_processor = Sam3Processor(sam3_model, confidence_threshold=0.3)
print("[Model Load] SAM 3 로딩 완료!")

# ---- [3] IDM-VTON Pipeline ----
print("[Model Load] IDM-VTON 파이프라인 로딩...")

unet = UNet2DConditionModel.from_pretrained(
    IDM_VTON_MODEL_PATH, subfolder="unet", torch_dtype=weight_dtype,
)
unet.requires_grad_(False)

unet_encoder = UNet2DConditionModel_ref.from_pretrained(
    IDM_VTON_MODEL_PATH, subfolder="unet_encoder", torch_dtype=weight_dtype,
)
unet_encoder.requires_grad_(False)

vae = AutoencoderKL.from_pretrained(
    IDM_VTON_MODEL_PATH, subfolder="vae", torch_dtype=weight_dtype,
)
vae.requires_grad_(False)

noise_scheduler = DDPMScheduler.from_pretrained(
    IDM_VTON_MODEL_PATH, subfolder="scheduler",
)

text_encoder_one = CLIPTextModel.from_pretrained(
    IDM_VTON_MODEL_PATH, subfolder="text_encoder", torch_dtype=weight_dtype,
)
text_encoder_one.requires_grad_(False)

text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    IDM_VTON_MODEL_PATH, subfolder="text_encoder_2", torch_dtype=weight_dtype,
)
text_encoder_two.requires_grad_(False)

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    IDM_VTON_MODEL_PATH, subfolder="image_encoder", torch_dtype=weight_dtype,
)
image_encoder.requires_grad_(False)

tokenizer_one = AutoTokenizer.from_pretrained(
    IDM_VTON_MODEL_PATH, subfolder="tokenizer", revision=None, use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    IDM_VTON_MODEL_PATH, subfolder="tokenizer_2", revision=None, use_fast=False,
)

# TryonPipeline 조립
pipe = TryonPipeline.from_pretrained(
    IDM_VTON_MODEL_PATH,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=weight_dtype,
).to(device)

pipe.unet_encoder = unet_encoder.to(device, weight_dtype)

print("[Model Load] IDM-VTON 파이프라인 로딩 완료!")

# 텐서 변환 (IDM-VTON 입력 전처리용)
tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

print("=" * 60)
print("[Worker Init] ✅ 모든 AI 모델 로딩 완료!")
print("=" * 60)

# -------------------------------------------------------------------
# Celery 초기화
# -------------------------------------------------------------------
REDIS_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis-queue:6379/0")
SPRING_BOOT_WEBHOOK_URL = os.getenv(
    "WEBHOOK_URL", "http://192.168.0.5:8080/api/v1/vto/complete"
)

app = Celery("vto_worker", broker=REDIS_URL, backend=REDIS_URL)


# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------
def download_image(url: str) -> Image.Image:
    """URL에서 이미지를 다운로드하여 PIL Image로 메모리에 반환합니다."""
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    return img


def send_webhook(
    job_id: str,
    status: str,
    result_image: Image.Image = None,
    error_message: str = None,
):
    """Spring Boot 웹훅으로 최종 결과물 또는 실패 상태를 Multipart로 전송합니다."""
    data = {"jobId": job_id, "status": status}
    files = None

    if error_message:
        data["errorMessage"] = error_message

    if result_image and status == "SUCCESS":
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format="JPEG", quality=95)
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
    """1악장: 옷 이미지 배경 제거 (rembg)"""
    print("[1악장] rembg로 옷 배경 제거 중...")
    result = rembg.remove(garment_image, session=rembg_session)
    # RGBA → RGB 변환 (흰색 배경)
    if result.mode == "RGBA":
        background = Image.new("RGB", result.size, (255, 255, 255))
        background.paste(result, mask=result.split()[3])
        result = background
    print("[1악장] 옷 배경 제거 완료!")
    return result


def extract_clothing_mask(human_image: Image.Image) -> Image.Image:
    """2악장: SAM 3 텍스트 프롬프트로 옷 영역 마스크 추출"""
    print("[2악장] SAM 3로 옷 영역 마스크 추출 중...")

    # SAM 3 추론
    inference_state = sam3_processor.set_image(human_image)
    inference_state = sam3_processor.set_text_prompt(
        state=inference_state, prompt="upper body clothing"
    )

    # 마스크 결과 추출
    masks = inference_state.get("masks", [])
    
    if len(masks) == 0:
        # 마스크를 찾지 못한 경우 전체 영역 마스크 반환
        print("[2악장] 마스크를 찾지 못했습니다. 전체 영역 사용.")
        mask = Image.new("L", human_image.size, 255)
    else:
        # 가장 큰 마스크를 선택하여 이진 마스크로 변환
        combined_mask = np.zeros(
            (human_image.size[1], human_image.size[0]), dtype=np.uint8
        )
        for m in masks:
            mask_np = np.array(m)
            if mask_np.ndim == 3:
                mask_np = mask_np[:, :, 0]
            combined_mask = np.maximum(combined_mask, (mask_np > 0).astype(np.uint8) * 255)
        mask = Image.fromarray(combined_mask, mode="L")

    print("[2악장] 옷 영역 마스크 추출 완료!")
    return mask


def run_idm_vton(
    human_image: Image.Image,
    garment_image: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """3악장: IDM-VTON 메인 인페인팅 합성"""
    print("[3악장] IDM-VTON 인페인팅 합성 시작...")

    # 이미지 리사이즈 (IDM-VTON 기본 해상도: 768x1024)
    target_w, target_h = 768, 1024
    human_resized = human_image.resize((target_w, target_h))
    garment_resized = garment_image.resize((target_w, target_h))
    mask_resized = mask.resize((target_w, target_h))

    # 마스크 전처리 (IDM-VTON에서는 inpaint_mask = 1 - mask)
    mask_tensor = transforms.ToTensor()(mask_resized)
    mask_tensor = mask_tensor[:1]  # 단일 채널
    inpaint_mask = mask_tensor  # 흰색(255)=인페인팅 영역

    # 마스크 적용된 이미지
    human_tensor = tensor_transform(human_resized)
    im_mask = human_tensor * (1 - inpaint_mask)

    # 옷 텍스트 설명 (기본값)
    garment_des = "upper body garment"

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            # 프롬프트 인코딩 (사람 + 옷)
            prompt = "model is wearing " + garment_des
            negative_prompt = (
                "monochrome, lowres, bad anatomy, worst quality, low quality, "
                "extra limbs, mutated hands, artifacts, deformed, missing limbs, fused fingers"
            )

            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            # 옷 전용 프롬프트 인코딩
            prompt_cloth = "a photo of " + garment_des
            if not isinstance(prompt_cloth, List):
                prompt_cloth = [prompt_cloth]

            (prompt_embeds_c, _, _, _) = pipe.encode_prompt(
                prompt_cloth,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=[negative_prompt],
            )

            # 텐서 준비
            # pose_img: SAM 3 마스크에서 유도된 간단한 마스크 이미지 (DensePose 대체)
            # IDM-VTON은 pose_img를 참고하지만 마스크로도 동작 가능
            pose_tensor = (
                tensor_transform(mask_resized.convert("RGB"))
                .unsqueeze(0)
                .to(device, weight_dtype)
            )
            garm_tensor = (
                tensor_transform(garment_resized)
                .unsqueeze(0)
                .to(device, weight_dtype)
            )

            generator = torch.Generator(device).manual_seed(42)

            images = pipe(
                prompt_embeds=prompt_embeds.to(device, weight_dtype),
                negative_prompt_embeds=negative_prompt_embeds.to(device, weight_dtype),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, weight_dtype),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(
                    device, weight_dtype
                ),
                num_inference_steps=30,
                generator=generator,
                strength=1.0,
                pose_img=pose_tensor,
                text_embeds_cloth=prompt_embeds_c.to(device, weight_dtype),
                cloth=garm_tensor,
                mask_image=inpaint_mask,
                image=human_resized,
                height=target_h,
                width=target_w,
                ip_adapter_image=garment_resized.resize((target_w, target_h)),
                guidance_scale=2.0,
            )[0]

    result = images[0]
    # 원본 사이즈로 복원
    result = result.resize(human_image.size)

    print("[3악장] IDM-VTON 인페인팅 합성 완료!")
    return result


# -------------------------------------------------------------------
# Main Celery Task
# -------------------------------------------------------------------
@app.task(name="process_vto", bind=True)
def process_vto(self, job_id: str, human_image_url: str, garment_image_url: str):
    """5단계 핵심 VTO 파이프라인 제어 및 실행 함수"""
    try:
        print(f"\n{'='*60}")
        print(f"[Pipeline] 작업 시작! Job ID: {job_id}")
        print(f"{'='*60}")

        # 입력 이미지 다운로드 (메모리)
        human_image = download_image(human_image_url)
        garment_image = download_image(garment_image_url)
        print(f"[Pipeline] 이미지 다운로드 완료 (사람: {human_image.size}, 옷: {garment_image.size})")

        # ============================================================
        # 1악장 & 2악장: 병렬 처리 (ThreadPoolExecutor)
        # ============================================================
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_garment = executor.submit(process_garment, garment_image)
            future_mask = executor.submit(extract_clothing_mask, human_image)

            processed_garment = future_garment.result()
            human_mask = future_mask.result()

        # ============================================================
        # 3악장: 메인 인페인팅 합성 (IDM-VTON)
        # ============================================================
        vto_result = run_idm_vton(human_image, processed_garment, human_mask)

        # ============================================================
        # 4악장: 안면 복원 (비활성 - 필요시 추후 활성화)
        # ============================================================
        final_image = vto_result

        # ============================================================
        # 5악장: Spring Boot 서버로 Multipart 웹훅 직전송
        # ============================================================
        send_webhook(job_id=job_id, status="SUCCESS", result_image=final_image)

        print(f"[Pipeline] ✅ 작업 완료! Job ID: {job_id}")
        return {"job_id": job_id, "status": "SUCCESS"}

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"[Worker Error] Job {job_id} failed:\n{error_msg}")

        # 에러 핸들링: Worker는 종료되지 않고 FAILED 상태와 에러 메시지를 서버로 회신
        send_webhook(job_id=job_id, status="FAILED", error_message=str(e))
        return {"job_id": job_id, "status": "FAILED", "error": str(e)}
