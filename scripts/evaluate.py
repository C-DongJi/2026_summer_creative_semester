"""재학습 효과 평가 스크립트 — 여러 모델 × 여러 평가셋의 SDR 비교.

평가셋은 MSST 레이아웃(<곡>/{vocals.wav, other.wav})이어야 하며, mixture는
스템 합으로 재구성해 각 모델로 분리한 뒤 정답 스템과의 SDR을 계산한다.
모델이 정확히 2개면 트랙별 페어드 비교(Δ, 승/무/패)까지 출력한다.

사용 예 (재학습 장르 + 타 장르 동시 평가):
    python scripts/evaluate.py \
      --eval-set pop=data/eval/pop --eval-set others=data/eval/others \
      --model base=models/checkpoints/MelBandRoformer.ckpt \
      --model pop_ft=models/checkpoints/finetune_pop/model.ckpt \
      --output outputs/eval_results.json

설계 근거와 판정 기준: docs/EVALUATION.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.audio.io import load_audio  # noqa: E402
from src.inference.pipeline import SeparationPipeline  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.metrics import sdr  # noqa: E402


def parse_labeled(items: list[str], flag: str) -> dict[str, str]:
    """["이름=값", ...] -> {이름: 값}. 형식 오류 시 명확히 실패."""
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"{flag} 형식은 '이름=경로' 입니다: {item!r}")
        name, _, value = item.partition("=")
        if not name or not value:
            raise SystemExit(f"{flag} 형식은 '이름=경로' 입니다: {item!r}")
        if name in out:
            raise SystemExit(f"{flag} 이름 중복: {name!r}")
        out[name] = value
    return out


def list_tracks(dataset_dir: Path) -> list[Path]:
    tracks = sorted(
        p for p in dataset_dir.iterdir()
        if p.is_dir() and (p / "vocals.wav").exists() and (p / "other.wav").exists()
    )
    if not tracks:
        raise SystemExit(f"평가 트랙이 없습니다: {dataset_dir}")
    return tracks


def evaluate_model(
    ckpt_path: str, eval_sets: dict[str, list[Path]],
    cfg_path: str, sr: int, ch: int, max_tracks: int | None,
) -> list[dict]:
    """한 모델로 모든 평가셋을 순회하며 트랙별 SDR 행을 만든다."""
    cfg = load_config(cfg_path)
    cfg["model"]["checkpoint"] = ckpt_path
    pipeline = SeparationPipeline(cfg)

    rows: list[dict] = []
    for set_name, tracks in eval_sets.items():
        subset = tracks[:max_tracks] if max_tracks else tracks
        for track in subset:
            ref_v, _ = load_audio(track / "vocals.wav", sr, ch)
            ref_o, _ = load_audio(track / "other.wav", sr, ch)
            n = min(ref_v.shape[-1], ref_o.shape[-1])
            ref_v, ref_o = ref_v[..., :n], ref_o[..., :n]
            mixture = ref_v + ref_o

            t0 = time.time()
            est = pipeline.separate_waveform(mixture)
            rows.append({
                "eval_set": set_name,
                "track": track.name,
                "sdr_vocals": round(sdr(ref_v, est["vocals"]), 4),
                "sdr_inst": round(sdr(ref_o, est["instrumental"]), 4),
                "seconds": round(time.time() - t0, 1),
            })
            r = rows[-1]
            print(f"  [{set_name}] {track.name}: vocals {r['sdr_vocals']:+.2f} dB / "
                  f"inst {r['sdr_inst']:+.2f} dB ({r['seconds']}s)")

    # 모델 교체 전 메모리 해제
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def summarize(results: dict[str, list[dict]]) -> None:
    """모델×평가셋 평균 표 + (2모델이면) 페어드 비교 출력."""
    models = list(results)
    set_names = sorted({r["eval_set"] for rows in results.values() for r in rows})

    print("\n=== 평균 SDR (dB) ===")
    header = f"{'평가셋':<12}" + "".join(f"{m:>22}" for m in models)
    print(header + ("  (vocals / inst)"))
    for s in set_names:
        cells = []
        for m in models:
            rs = [r for r in results[m] if r["eval_set"] == s]
            mv = sum(r["sdr_vocals"] for r in rs) / len(rs)
            mi = sum(r["sdr_inst"] for r in rs) / len(rs)
            cells.append(f"{mv:+.2f} / {mi:+.2f}".rjust(22))
        print(f"{s:<12}" + "".join(cells))

    if len(models) == 2:
        base, ft = models
        print(f"\n=== 페어드 비교: {ft} − {base} (vocals SDR) ===")
        for s in set_names:
            b = {r["track"]: r for r in results[base] if r["eval_set"] == s}
            f = {r["track"]: r for r in results[ft] if r["eval_set"] == s}
            common = sorted(set(b) & set(f))
            deltas = [f[t]["sdr_vocals"] - b[t]["sdr_vocals"] for t in common]
            wins = sum(d > 0 for d in deltas)
            mean = sum(deltas) / len(deltas)
            print(f"  {s}: 평균 Δ {mean:+.2f} dB | 향상 {wins}/{len(deltas)}곡 | "
                  f"최대 {max(deltas):+.2f} / 최소 {min(deltas):+.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="모델×평가셋 SDR 비교 평가")
    parser.add_argument("--eval-set", action="append", required=True,
                        help="이름=디렉토리 (반복 지정). 예: pop=data/eval/pop")
    parser.add_argument("--model", action="append", required=True,
                        help="이름=체크포인트 (반복 지정). 예: base=models/.../MelBandRoformer.ckpt")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default=None, help="트랙별 결과 JSON 저장 경로")
    parser.add_argument("--max-tracks", type=int, default=None,
                        help="평가셋당 최대 트랙 수 (스모크 테스트용)")
    args = parser.parse_args()

    eval_dirs = parse_labeled(args.eval_set, "--eval-set")
    models = parse_labeled(args.model, "--model")
    cfg = load_config(args.config)
    sr, ch = cfg.audio.sample_rate, cfg.audio.channels

    eval_sets = {name: list_tracks(Path(d)) for name, d in eval_dirs.items()}
    for name, tracks in eval_sets.items():
        print(f"평가셋 '{name}': {len(tracks)}곡")

    results: dict[str, list[dict]] = {}
    for m_name, ckpt in models.items():
        print(f"\n### 모델 '{m_name}' ({ckpt}) 평가 중...")
        results[m_name] = evaluate_model(
            ckpt, eval_sets, args.config, sr, ch, args.max_tracks
        )

    summarize(results)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n트랙별 결과 저장: {out}")


if __name__ == "__main__":
    main()
