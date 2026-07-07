#!/usr/bin/env bash
# MSST 프레임워크 체크아웃 + Kim Mel-Band RoFormer 체크포인트 다운로드
# 사용: bash scripts/setup_msst.sh   (프로젝트 루트에서 실행)
set -euo pipefail

MSST_DIR="third_party/Music-Source-Separation-Training"
CKPT="models/checkpoints/MelBandRoformer.ckpt"

mkdir -p third_party models/checkpoints

if [ ! -d "$MSST_DIR" ]; then
  echo "[1/3] MSST 프레임워크 클론 (MIT)..."
  git clone --depth 1 https://github.com/ZFTurbo/Music-Source-Separation-Training.git "$MSST_DIR"
else
  echo "[1/3] MSST 이미 존재 — 건너뜀 ($MSST_DIR)"
fi

if [ ! -f "$CKPT" ]; then
  echo "[2/3] Kim Mel-Band RoFormer 체크포인트 다운로드 (MIT, ~913MB)..."
  wget -O "$CKPT" \
    "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt"
else
  echo "[2/3] 체크포인트 이미 존재 — 건너뜀 ($CKPT)"
fi

echo "[3/3] MSST 의존성 설치..."
pip install -r "$MSST_DIR/requirements.txt"

echo "완료. 다음으로:"
echo "  추론 테스트:  python scripts/separate.py --input <곡.mp3> --output outputs/"
echo "  파인튜닝:     docs/PIPELINE_DESIGN.md Mode 2 참고"
