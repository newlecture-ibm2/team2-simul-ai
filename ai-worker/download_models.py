"""
모델 가중치 사전 다운로드 스크립트

Docker 이미지 빌드 시 또는 컨테이너 최초 실행 시 
모델 가중치를 로컬에 캐싱하기 위한 스크립트입니다.

사용법:
    python download_models.py

주의: HuggingFace 인증이 필요한 경우 
      huggingface-cli login 을 먼저 실행하세요.
"""
import os


def download_idm_vton():
    """IDM-VTON 모델 가중치를 HuggingFace에서 다운로드합니다."""
    from huggingface_hub import snapshot_download
    
    cache_dir = os.getenv("MODEL_CACHE_DIR", "/app/models")
    model_id = "yisol/IDM-VTON"
    
    print(f"[Download] IDM-VTON 모델 다운로드 시작: {model_id}")
    snapshot_download(
        repo_id=model_id,
        local_dir=os.path.join(cache_dir, "idm-vton"),
        local_dir_use_symlinks=False,
    )
    print("[Download] IDM-VTON 모델 다운로드 완료!")


def download_sam3():
    """SAM 3 모델 체크포인트를 다운로드합니다."""
    from huggingface_hub import hf_hub_download
    
    cache_dir = os.getenv("MODEL_CACHE_DIR", "/app/models")
    sam3_dir = os.path.join(cache_dir, "sam3")
    os.makedirs(sam3_dir, exist_ok=True)
    
    print("[Download] SAM 3 체크포인트 다운로드 시작...")
    # SAM 3 large 모델 다운로드
    hf_hub_download(
        repo_id="facebook/sam3-hiera-large",
        filename="sam3_hiera_large.pt",
        local_dir=sam3_dir,
    )
    print("[Download] SAM 3 체크포인트 다운로드 완료!")


def download_rembg():
    """rembg 모델을 미리 다운로드합니다."""
    import rembg
    
    print("[Download] rembg (u2net) 모델 다운로드 시작...")
    rembg.new_session("u2net")
    print("[Download] rembg 모델 다운로드 완료!")


if __name__ == "__main__":
    download_rembg()
    download_sam3()
    download_idm_vton()
    print("\n✅ 모든 모델 가중치 다운로드 완료!")
