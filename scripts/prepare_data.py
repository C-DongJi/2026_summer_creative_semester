"""CLI 데이터 전처리 진입점.

사용 예:
    python scripts/prepare_data.py --config config/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.preprocess import preprocess_dataset  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="커스텀 데이터 전처리")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--raw", default=None, help="raw 디렉토리 (설정 덮어쓰기)")
    parser.add_argument("--out", default=None, help="processed 디렉토리 (설정 덮어쓰기)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw = args.raw or cfg.data.raw_dir
    out = args.out or cfg.data.processed_dir

    count = preprocess_dataset(
        raw_dir=raw,
        processed_dir=out,
        sample_rate=cfg.audio.sample_rate,
        channels=cfg.audio.channels,
        normalize=cfg.data.normalize,
    )
    print(f"전처리 완료: {count}개 트랙 -> {out}")


if __name__ == "__main__":
    main()
