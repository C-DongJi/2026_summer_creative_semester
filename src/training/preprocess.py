"""커스텀 데이터 전처리 모듈 (Mode 2 입구, 담당: 안준석).

운영자가 수집한 (원본 Mix, 반주 Inst) 쌍을 MSST dataset_type 4 레이아웃의
학습용 데이터셋으로 변환한다:

입력:  data/raw/<곡명>/{mix.*, inst.*}
출력:  data/processed/<곡명>/{vocals.wav, other.wav}
        - other.wav  = 반주 (공통 게인 적용)
        - vocals.wav = mix − inst (float32 감산 후 32-bit float WAV 저장)

핵심 주의사항 (docs/PIPELINE_DESIGN.md §4):
  1. mix/inst에 '동일한' 게인만 적용 — 개별 정규화하면 감산 관계가 깨져
     보컬 스템에 반주가 새어 들어간다.
  2. 감산 전 샘플 정렬/길이 검증 필수. 길이 차이가 크면 인코딩 오프셋
     의심 → 해당 쌍 스킵하고 경고.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import torch

from src.audio.io import load_audio, save_audio

# 이 이상 길이가 어긋나면 단순 트림으로 해결될 문제가 아니라고 보고 스킵 (초)
MAX_LENGTH_MISMATCH_SEC = 0.5


def find_pairs(raw_dir: str | Path) -> list[tuple[Path, Path]]:
    """raw 디렉토리에서 (mix, inst) 파일 쌍을 찾는다."""
    raw_dir = Path(raw_dir)
    pairs: list[tuple[Path, Path]] = []
    for track in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        mix = next(track.glob("mix.*"), None)
        inst = next(track.glob("inst.*"), None)
        if mix and inst:
            pairs.append((mix, inst))
        else:
            warnings.warn(f"mix/inst 쌍이 불완전하여 건너뜀: {track.name}")
    return pairs


def derive_stems(
    mix: torch.Tensor,
    inst: torch.Tensor,
    sample_rate: int,
    normalize: bool = True,
    target_peak: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(mix, inst) -> (vocals, other). 정렬 실패 시 None.

    vocals = mix − inst. 게인은 mix 피크 기준으로 계산해 양쪽에 동일 적용.
    """
    len_mix, len_inst = mix.shape[-1], inst.shape[-1]
    mismatch_sec = abs(len_mix - len_inst) / sample_rate
    if mismatch_sec > MAX_LENGTH_MISMATCH_SEC:
        return None

    length = min(len_mix, len_inst)
    mix, inst = mix[..., :length], inst[..., :length]

    if normalize:
        peak = mix.abs().max()
        gain = (target_peak / peak) if peak > 1e-8 else 1.0
        mix, inst = mix * gain, inst * gain

    vocals = mix - inst
    return vocals, inst


def preprocess_dataset(
    raw_dir: str | Path,
    processed_dir: str | Path,
    sample_rate: int = 44100,
    channels: int = 2,
    normalize: bool = True,
    target_peak: float = 0.95,
) -> int:
    """raw의 모든 (mix, inst) 쌍을 MSST 레이아웃으로 변환한다.

    Returns:
        처리한 트랙 수
    """
    processed_dir = Path(processed_dir)
    count = 0
    for mix_path, inst_path in find_pairs(raw_dir):
        track = mix_path.parent.name
        mix, _ = load_audio(mix_path, sample_rate, channels)
        inst, _ = load_audio(inst_path, sample_rate, channels)

        stems = derive_stems(mix, inst, sample_rate, normalize, target_peak)
        if stems is None:
            warnings.warn(
                f"{track}: mix/inst 길이 차이가 {MAX_LENGTH_MISMATCH_SEC}s 초과 — "
                "인코딩 오프셋 의심, 건너뜀"
            )
            continue
        vocals, other = stems

        out_dir = processed_dir / track
        # 감산으로 유도한 스템은 양자화 손실을 피하려 32-bit float로 저장
        save_audio(out_dir / "vocals.wav", vocals, sample_rate, float32=True)
        save_audio(out_dir / "other.wav", other, sample_rate, float32=True)
        count += 1
    return count
