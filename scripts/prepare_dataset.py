"""공개 멀티 스템 데이터셋 → 2-stem 학습셋 변환 진입점.

[창의학기제 5주차 (7/20) 산출물]

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
    parser.add_argument("--src", default=None, help="MoisesDB 데이터 경로 (기본: config public_datasets.moisesdb_dir)")
    parser.add_argument("--out", default=None, help="학습셋 출력 (기본: config data.processed_dir)")
    parser.add_argument("--valid-dir", default=None, help="검증셋 출력 (기본: config data.valid_dir)")
    parser.add_argument("--holdout-frac", type=float, default=None,
                        help="검증셋 비율 (기본: config public_datasets.holdout_frac)")
    parser.add_argument("--versions", nargs="+", default=None,
                        help="MedleyDB 버전 (기본: config public_datasets.medleydb_versions)")
    parser.add_argument("--include-speech", action="store_true", help="MedleyDB: speaker/crowd도 보컬로")
    parser.add_argument("--keep-bleed", action="store_true", help="MedleyDB: has_bleed 트랙도 포함")
    parser.add_argument("--genres", nargs="+", default=None,
                        help="지정 장르 트랙만 변환 (부분 일치). 예: --genres rock pop")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pub = getattr(cfg, "public_datasets", None)
    sr = cfg.audio.sample_rate
    channels = cfg.audio.channels
    out_dir = Path(args.out or cfg.data.processed_dir)
    valid_dir = Path(args.valid_dir or cfg.data.valid_dir)
    # CLI 미지정 시 config public_datasets 섹션의 기본값 사용
    if args.src is None and pub is not None:
        args.src = pub.moisesdb_dir
    if args.versions is None:
        args.versions = list(pub.medleydb_versions) if pub is not None else ["V1", "V2"]
    if args.holdout_frac is None:
        args.holdout_frac = pub.holdout_frac if pub is not None else 0.1
    include_speech = args.include_speech or bool(
        pub is not None and pub.medleydb_include_speech
    )
    drop_bleed = (not args.keep_bleed) and bool(
        pub is None or pub.medleydb_drop_bleed
    )

    if args.dataset == "moisesdb":
        if not args.src:
            parser.error("--dataset moisesdb 는 --src (데이터 경로)가 필요합니다.")
        from src.data.moisesdb_ingest import iter_moisesdb_pairs
        pairs = iter_moisesdb_pairs(
            args.src, sample_rate=sr, channels=channels, genres=args.genres
        )
    else:
        from src.data.medleydb_ingest import iter_medleydb_pairs
        pairs = iter_medleydb_pairs(
            versions=args.versions,
            include_speech=include_speech,
            drop_bleed=drop_bleed,
            sample_rate=sr,
            channels=channels,
            genres=args.genres,
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
