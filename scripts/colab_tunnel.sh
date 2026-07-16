#!/usr/bin/env bash
# Colab "터미널"에서 VS Code 터널을 여는 원커맨드 스크립트.
# (노트북 셀이 아니라 터미널에서 실행 — 셀은 자유롭게 쓸 수 있게 됨)
#
# 사용:
#   bash scripts/colab_tunnel.sh            # 기본 터널명: colab-gpu
#   bash scripts/colab_tunnel.sh my-tunnel  # 터널명 지정
#
# 실행하면 github.com/login/device 주소와 8자리 코드가 출력된다.
# 그 주소에서 코드를 입력해 GitHub 인증하면 터널이 열린다.
# 이 프로세스는 포그라운드로 유지해야 한다 (끄면 터널이 닫힘).
set -euo pipefail

TUNNEL_NAME="${1:-colab-gpu}"
WORKDIR="${HOME}"
[ -d /content ] && WORKDIR=/content
cd "$WORKDIR"

if [ ! -x ./code ]; then
  echo "[1/2] VS Code CLI 다운로드..."
  curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' \
    --output vscode_cli.tar.gz
  tar -xf vscode_cli.tar.gz
else
  echo "[1/2] VS Code CLI 이미 존재 — 건너뜀"
fi

echo "[2/2] 터널 시작 (이름: ${TUNNEL_NAME}) — 이 창을 닫지 마세요"
exec ./code tunnel --accept-server-license-terms --name "$TUNNEL_NAME"
