"""공개 멀티 스템 데이터셋 → 2-stem 학습셋 변환 진입점.

MoisesDB / MedleyDB를 보컬/반주 2-stem으로 접어 MSST dataset_type 4
레이아웃으로 저장한다. 일부를 검증셋(valid)으로 결정적으로 분리한다.

사용 예:
    # MoisesDB
    python scripts/prepare_dataset.py --dataset moisesdb \
        --src /data/moisesdb_v0.1 --out data/processed --valid-dir data/valid

    # MedleyDB (MEDLEYDB_PATH 환경변수로 경로 지정)
    export MEDLEYDB_PATH=/data/MedleyDB
    python scripts/prepare_dataset.py --dataset medleydb \
        --out data/processed --valid-dir data/valid --versions V1 V2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingest_common import save_stem_pair  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="공개 데이터셋 -> 2-stem 학습셋 변환")
    parser.add_argument("--dataset", required=True, choices=["moisesdb", "medleydb"])
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--src", default=None, help="MoisesDB 데이터 경로 (medleydb는 MEDLEYDB_PATH 사용)")
    parser.add_argument("--out", default=None, help="학습셋 출력 (기본: config data.processed_dir)")
    parser.add_argument("--valid-dir", default=None, help="검증셋 출력 (기본: config data.valid_dir)")
    parser.add_argument("--holdout-frac", type=float, default=0.1, help="검증셋 비율 (결정적 분리)")
    parser.add_argument("--versions", nargs="+", default=["V1", "V2"], help="MedleyDB 버전")
    parser.add_argument("--include-speech", action="store_true", help="MedleyDB: speaker/crowd도 보컬로")
    parser.add_argument("--keep-bleed", action="store_true", help="MedleyDB: has_bleed 트랙도 포함")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sr = cfg.audio.sample_rate
    channels = cfg.audio.channels
    out_dir = Path(args.out or cfg.data.processed_dir)
    valid_dir = Path(args.valid_dir or cfg.data.valid_dir)

    if args.dataset == "moisesdb":
        if not args.src:
            parser.error("--dataset moisesdb 는 --src (데이터 경로)가 필요합니다.")
        from src.data.moisesdb_ingest import iter_moisesdb_pairs
        pairs = iter_moisesdb_pairs(args.src, sample_rate=sr, channels=channels)
    else:
        from src.data.medleydb_ingest import iter_medleydb_pairs
        pairs = iter_medleydb_pairs(
            versions=args.versions,
            include_speech=args.include_speech,
            drop_bleed=not args.keep_bleed,
            sample_rate=sr,
            channels=channels,
        )

    # 결정적 검증셋 분리: holdout_frac 비율마다 1곡을 valid로
    every = max(2, round(1 / args.holdout_frac)) if args.holdout_frac > 0 else 0
    n_train = n_valid = 0
    for i, (name, vocals, other) in enumerate(pairs):
        is_valid = every and (i % every == 0)
        dest = (valid_dir if is_valid else out_dir) / name
        save_stem_pair(dest, vocals, other, sr)
        if is_valid:
            n_valid += 1
        else:
            n_train += 1
        total = n_train + n_valid
        if total % 10 == 0:
            print(f"  ... {total}곡 처리 (train {n_train} / valid {n_valid})")

    print(f"\n완료: train {n_train}곡 -> {out_dir}")
    print(f"      valid {n_valid}곡 -> {valid_dir}")
    print("검증:  python scripts/check_dataset.py --dir", out_dir)


if __name__ == "__main__":
    main()
