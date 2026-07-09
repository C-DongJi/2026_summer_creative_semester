"""2주차 + 3주차 검증 스크립트.

동일한 긴 음원을 두 방식으로 추론하며 GPU 피크 VRAM과 소요 시간을 비교한다:
  (A) 베이스라인 — 전체를 한 번에 모델에 투입 (청크 없음)  → 2주차 OOM 관찰
  (B) 청크 분할 추론 + Overlap-Add                          → 3주차 알고리즘 실증

베이스라인의 VRAM은 음원 길이에 비례해 커지고(길면 OOM),
청크 방식은 청크 하나 크기로 고정됨을 보인다.

사용 예 (Colab GPU 런타임):
    # 합성 노이즈 4분으로 측정
    python scripts/verify_oom_chunking.py --minutes 4
    # 실제 곡으로 측정
    python scripts/verify_oom_chunking.py --input song.mp3
    # 더 긴 길이로 베이스라인 OOM 유도
    python scripts/verify_oom_chunking.py --minutes 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.audio.chunking import chunked_inference  # noqa: E402
from src.audio.io import load_audio  # noqa: E402
from src.models.registry import load_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.device import get_device  # noqa: E402


def _measure(fn, device: torch.device) -> dict:
    """fn 실행 중 피크 VRAM(GB)과 소요 시간(s)을 측정. OOM은 잡아서 보고."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    status = "ok"
    try:
        fn()
    except torch.cuda.OutOfMemoryError:
        status = "OOM"
    except RuntimeError as exc:  # CPU OOM 등은 RuntimeError로 옴
        status = "OOM" if "out of memory" in str(exc).lower() else f"error: {exc}"
    dt = time.time() - t0
    peak = (
        torch.cuda.max_memory_allocated(device) / 1e9
        if device.type == "cuda"
        else float("nan")
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"status": status, "seconds": dt, "peak_vram_gb": peak}


def main() -> None:
    parser = argparse.ArgumentParser(description="OOM vs 청크 추론 검증")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--input", default=None, help="실제 오디오 파일 (없으면 합성 노이즈)")
    parser.add_argument("--minutes", type=float, default=4.0, help="합성 노이즈 길이(분)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.inference.device)
    sr = cfg.audio.sample_rate

    print(f"[env] device={device}", end="")
    if device.type == "cuda":
        print(f" | {torch.cuda.get_device_name(device)} | "
              f"total {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    else:
        print(" | (CPU: VRAM OOM은 재현되지 않음 — GPU 런타임에서 실행하세요)")

    print(f"[model] loading {cfg.model.name} ...")
    model = load_model(cfg.model, device=device)

    # 입력 음원 준비
    if args.input:
        mixture, _ = load_audio(args.input, sr, cfg.audio.channels)
        dur = mixture.shape[-1] / sr
        print(f"[audio] input={args.input} | {dur:.1f}s ({dur/60:.1f}min)")
    else:
        n = int(args.minutes * 60 * sr)
        mixture = torch.randn(cfg.audio.channels, n) * 0.1
        print(f"[audio] synthetic noise | {args.minutes:.1f}min ({n:,} samples)")

    def process_fn(chunk: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = model(chunk.unsqueeze(0))
        return out.squeeze(0)

    def run_baseline() -> None:
        with torch.no_grad():
            _ = model(mixture.unsqueeze(0).to(device))

    def run_chunked() -> None:
        _ = chunked_inference(
            mixture,
            process_fn=process_fn,
            chunk_samples=int(cfg.inference.chunk_seconds * sr),
            overlap=cfg.inference.overlap,
            fade=cfg.inference.fade,
            device=device,
        )

    print("\n[A] 베이스라인 (청크 없음, 전체 일괄 추론) ...")
    a = _measure(run_baseline, device)
    print(f"    → {a['status']} | {a['seconds']:.1f}s | peak VRAM {a['peak_vram_gb']:.2f} GB")

    print(f"[B] 청크 분할 추론 (chunk {cfg.inference.chunk_seconds}s, "
          f"overlap {cfg.inference.overlap}) ...")
    b = _measure(run_chunked, device)
    print(f"    → {b['status']} | {b['seconds']:.1f}s | peak VRAM {b['peak_vram_gb']:.2f} GB")

    print("\n=== 요약 ===")
    print(f"  베이스라인 : {a['status']:>5} | peak {a['peak_vram_gb']:.2f} GB")
    print(f"  청크 방식  : {b['status']:>5} | peak {b['peak_vram_gb']:.2f} GB")
    if device.type == "cuda" and a["status"] == "ok" and b["status"] == "ok":
        saved = a["peak_vram_gb"] - b["peak_vram_gb"]
        print(f"  VRAM 절감  : {saved:.2f} GB ({saved / a['peak_vram_gb'] * 100:.0f}%)")
    if a["status"] == "OOM" and b["status"] == "ok":
        print("  ✅ 베이스라인은 OOM, 청크 방식은 성공 — 3주차 알고리즘의 효과 실증")


if __name__ == "__main__":
    main()
