"""전처리된 2-stem 학습셋 검증·통계 스크립트.

MSST dataset_type 4 레이아웃(<곡>/{vocals.wav, other.wav})을 스캔해
학습에 문제가 될 만한 것들을 점검하고 통계를 출력한다.

점검 항목:
  - vocals.wav / other.wav 존재, 길이·샘플레이트·채널 일치
  - NaN/Inf 여부
  - vocals가 사실상 무음인지 (RMS 임계 미만 → 반주 트랙이 섞였을 가능성)
  - mixture(vocals+other) 클리핑 (|peak| > 1.0)
  - 트랙 길이 분포, 총 시간

사용 예:
    python scripts/check_dataset.py --dir data/processed
    python scripts/check_dataset.py --dir data/valid --verbose
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

SILENCE_RMS = 1e-4   # 이 미만이면 사실상 무음으로 간주


def _load(path: Path) -> tuple[torch.Tensor, int]:
    """원본 샘플레이트·길이 그대로 로드 (리샘플/채널변환 없이 점검용)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (frames, channels)
    return torch.from_numpy(data.T).contiguous(), sr  # (channels, frames)


def check_track(track_dir: Path) -> dict:
    """트랙 하나를 점검해 결과 dict 반환."""
    r: dict = {"name": track_dir.name, "issues": [], "duration": 0.0}
    vpath, opath = track_dir / "vocals.wav", track_dir / "other.wav"
    if not vpath.exists() or not opath.exists():
        r["issues"].append("missing_stem")
        return r

    vocals, vsr = _load(vpath)
    other, osr = _load(opath)

    if vsr != osr:
        r["issues"].append(f"sr_mismatch({vsr}!={osr})")
    if vocals.shape[0] != other.shape[0]:
        r["issues"].append(f"ch_mismatch({vocals.shape[0]}!={other.shape[0]})")
    if vocals.shape[-1] != other.shape[-1]:
        r["issues"].append(
            f"len_mismatch({vocals.shape[-1]}!={other.shape[-1]})"
        )

    r["sample_rate"] = vsr
    r["channels"] = vocals.shape[0]
    r["duration"] = vocals.shape[-1] / vsr

    if not torch.isfinite(vocals).all() or not torch.isfinite(other).all():
        r["issues"].append("nan_or_inf")

    v_rms = vocals.pow(2).mean().sqrt().item()
    r["vocals_rms"] = v_rms
    if v_rms < SILENCE_RMS:
        r["issues"].append("silent_vocals")

    # mixture 클리핑 (공통 길이 기준)
    n = min(vocals.shape[-1], other.shape[-1])
    mix_peak = (vocals[..., :n] + other[..., :n]).abs().max().item()
    r["mix_peak"] = mix_peak
    if mix_peak > 1.0:
        r["issues"].append(f"mix_clip({mix_peak:.2f})")

    return r


def main() -> None:
    parser = argparse.ArgumentParser(description="2-stem 학습셋 검증·통계")
    parser.add_argument("--dir", required=True, help="처리된 데이터셋 디렉토리")
    parser.add_argument("--verbose", action="store_true", help="이상 없는 트랙도 모두 출력")
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.exists():
        parser.error(f"디렉토리가 없습니다: {root}")

    tracks = sorted(p for p in root.iterdir() if p.is_dir())
    if not tracks:
        print(f"트랙이 없습니다: {root}")
        return

    results = [check_track(t) for t in tracks]

    total_dur = sum(r["duration"] for r in results)
    ok = [r for r in results if not r["issues"]]
    bad = [r for r in results if r["issues"]]
    durations = [r["duration"] for r in results if r["duration"] > 0]

    print(f"=== 데이터셋 점검: {root} ===")
    print(f"트랙 수      : {len(results)}  (정상 {len(ok)} / 이상 {len(bad)})")
    print(f"총 길이      : {total_dur / 3600:.2f} 시간")
    if durations:
        print(f"트랙 길이(초): min {min(durations):.0f} / "
              f"mean {sum(durations) / len(durations):.0f} / max {max(durations):.0f}")

    # 이슈 유형별 집계
    issue_counts: dict[str, int] = {}
    for r in bad:
        for iss in r["issues"]:
            key = iss.split("(")[0]
            issue_counts[key] = issue_counts.get(key, 0) + 1
    if issue_counts:
        print("\n이슈 요약:")
        for key, cnt in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  - {key}: {cnt}곡")

    if bad:
        print("\n이상 트랙:")
        for r in bad:
            print(f"  [{','.join(r['issues'])}]  {r['name']}")

    if args.verbose:
        print("\n전체 트랙:")
        for r in results:
            tag = "OK " if not r["issues"] else "!! "
            print(f"  {tag}{r['name']}  {r['duration']:.0f}s  "
                  f"vRMS={r.get('vocals_rms', float('nan')):.4f}  "
                  f"peak={r.get('mix_peak', float('nan')):.2f}")

    # 종료 코드: 이상 있으면 1
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
