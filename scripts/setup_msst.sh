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

echo "[3/3] MSST 의존성 설치 (torch 계열/GUI 패키지 제외)..."
# - torch/torchaudio: 그대로 설치하면 환경의 CUDA/CPU 빌드가 덮어써짐
# - wxpython/pyaudio: MSST GUI 도구 전용. 빌드가 자주 실패하며(휠 없음),
#   실패 시 pip이 전체 설치를 롤백하므로 학습/추론에 불필요한 이 둘도 제외
grep -viE '^\s*(torch|torchaudio|torchvision|wxpython|pyaudio)([=<>!~ ]|$)' \
  "$MSST_DIR/requirements.txt" > /tmp/msst_requirements_filtered.txt
pip install -r /tmp/msst_requirements_filtered.txt
# MSST가 구버전 librosa(0.9.x)를 고정하는데, 이는 setuptools 81+에서 제거된
# pkg_resources를 임포트한다 → setuptools를 81 미만으로 고정
pip install "setuptools<81"

echo "완료. 다음으로:"
echo "  추론 테스트:  python scripts/separate.py --input <곡.mp3> --output outputs/"
echo "  파인튜닝:     docs/PIPELINE_DESIGN.md Mode 2 참고"
