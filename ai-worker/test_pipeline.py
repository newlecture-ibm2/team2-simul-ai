"""
VTO AI Pipeline Standalone E2E Test Script
===========================================
이 스크립트는 Celery, Redis, Spring Boot 백엔드와 무관하게 
서버 단독으로 AI 파이프라인의 입출력을 테스트하기 위한 용도입니다.

사용법:
    docker compose exec ai-worker python test_pipeline.py --human <URL> --garment <URL>
"""
import os
import sys
import argparse

# 루트 및 tasks.py 모듈 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# tasks.py의 전역 모델 로드 및 함수 호출
from tasks import download_image, process_garment, extract_clothing_mask, run_idm_vton


def main():
    parser = argparse.ArgumentParser(description="VTO Standalone Pipeline Test")
    parser.add_argument(
        "--human",
        required=True,
        help="URL of human image to test",
    )
    parser.add_argument(
        "--garment",
        required=True,
        help="URL of garment image to test",
    )
    parser.add_argument(
        "--output",
        default="/app/output/test_result.jpg",
        help="Path to save the result image",
    )
    args = parser.parse_args()

    # 출력 폴더 생성 확인
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print("🚀 [TEST] Standalone E2E VTO 파이프라인 검증 시작")
    print("=" * 50)

    # 1. 이미지 다운로드
    print("\n[1단계] 이미지 다운로드 중...")
    try:
        human_img = download_image(args.human)
        garment_img = download_image(args.garment)
        print(f"  - 인물 이미지 크기: {human_img.size}")
        print(f"  - 의류 이미지 크기: {garment_img.size}")
    except Exception as e:
        print(f"❌ 이미지 다운로드 실패: {e}")
        return

    # 2. 옷 이미지 배경 제거 (1악장)
    print("\n[2단계] 옷 이미지 배경 제거 실행 (rembg)...")
    processed_garment = process_garment(garment_img)

    # 3. 옷 마스크 영역 추출 (2악장)
    print("\n[3단계] 인물 이미지에서 마스크 추출 실행 (SAM 3)...")
    mask = extract_clothing_mask(human_img)
    
    # 디버깅용 마스크 저장
    mask_path = args.output.replace(".jpg", "_mask.png")
    mask.save(mask_path)
    print(f"  - 디버깅용 마스크 이미지 저장 완료: {mask_path}")

    # 4. 가상 피팅 합성 (3악장)
    print("\n[4단계] 가상 피팅 합성 실행 (IDM-VTON)...")
    try:
        result = run_idm_vton(human_img, processed_garment, mask)
    except Exception as e:
        print(f"❌ 가상 피팅 합성 실패: {e}")
        return

    # 5. 결과 저장
    print(f"\n[5단계] 최종 결과 이미지 저장 중: {args.output}")
    result.save(args.output, format="JPEG", quality=95)
    
    print("\n" + "=" * 50)
    print("✅ [TEST] 모든 파이프라인 검증 완료!")
    print(f"👉 호스트 PC의 './output/' 폴더에서 파일을 확인하세요.")
    print("=" * 50)


if __name__ == "__main__":
    main()
