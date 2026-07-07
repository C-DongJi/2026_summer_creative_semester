"""CLI 추론 진입점.

사용 예:
    python scripts/separate.py --input song.mp3 --output outputs/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.pipeline import SeparationPipeline  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="음원 분리 추론")
    parser.add_argument("--input", required=True, help="입력 오디오 파일")
    parser.add_argument("--output", default="outputs", help="출력 디렉토리")
    parser.add_argument("--config", default="config/default.yaml", help="설정 파일")
    parser.add_argument("--model", default=None, help="모델 이름 (설정 덮어쓰기)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["name"] = args.model

    pipeline = SeparationPipeline(cfg)
    paths = pipeline.separate_to_files(args.input, args.output)
    print("분리 완료:")
    for name, path in paths.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
