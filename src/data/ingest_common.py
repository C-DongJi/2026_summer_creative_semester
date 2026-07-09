"""외부 멀티 스템 데이터셋 → 2-stem(보컬/반주) 변환 공통 유틸.

MoisesDB·MedleyDB는 악기별 멀티 스템 데이터셋이지만, 본 프로젝트는
보컬/반주 2-stem만 사용한다(악기별 supervision 없음). 따라서 각 트랙을
  vocals = 보컬 스템(들)의 합
  other  = 보컬 이외 모든 스템의 합
  mixture = vocals + other  (스템 합으로 자기정합적으로 정의)
로 접어서 MSST dataset_type 4 레이아웃(<곡>/{vocals.wav, other.wav})으로 저장한다.
"""
from __future__ import annotations

from pathlib import Path

import torch

from src.audio.io import save_audio

# MedleyDB 보컬 라벨 (taxonomy 'voices' 카테고리, 검증된 12종)
# 노래(가창) 분리 기준의 기본 집합 — 말하기(speaker)/군중(crowd)은 제외.
VOICE_LABELS_SINGING = {
    "male singer", "female singer", "vocalists", "choir",
    "male rapper", "female rapper", "beatboxing",
    "male screamer", "female screamer",
}
# 음성 전체를 보컬로 볼 때(구어 포함) 추가되는 라벨
VOICE_LABELS_SPEECH = {"male speaker", "female speaker", "crowd"}


def to_stereo_tensor(arr, channels: int = 2) -> torch.Tensor:
    """numpy/torch 오디오를 [channels, samples] float32 텐서로 정규화."""
    if not isinstance(arr, torch.Tensor):
        arr = torch.as_tensor(arr)
    arr = arr.float()
    if arr.dim() == 1:  # [samples] -> [1, samples]
        arr = arr.unsqueeze(0)
    cur = arr.shape[0]
    if cur == channels:
        return arr
    if cur == 1:
        return arr.repeat(channels, 1)
    if cur > channels:
        return arr[:channels]
    return arr.repeat(channels, 1)[:channels]


def pad_and_sum(tensors: list[torch.Tensor]) -> torch.Tensor | None:
    """[C, T] 텐서 리스트를 최장 길이에 맞춰 zero-pad 후 합산."""
    tensors = [t for t in tensors if t is not None]
    if not tensors:
        return None
    channels = tensors[0].shape[0]
    max_len = max(t.shape[-1] for t in tensors)
    acc = torch.zeros(channels, max_len, dtype=torch.float32)
    for t in tensors:
        acc[..., : t.shape[-1]] += t
    return acc


def save_stem_pair(
    out_dir: str | Path,
    vocals: torch.Tensor,
    other: torch.Tensor,
    sample_rate: int,
) -> None:
    """vocals/other를 공통 길이로 맞춰 32-bit float WAV로 저장."""
    max_len = max(vocals.shape[-1], other.shape[-1])
    vocals = _pad_to(vocals, max_len)
    other = _pad_to(other, max_len)
    out_dir = Path(out_dir)
    # 감산/합산으로 유도한 스템은 양자화 손실을 피하려 float32 저장
    save_audio(out_dir / "vocals.wav", vocals, sample_rate, float32=True)
    save_audio(out_dir / "other.wav", other, sample_rate, float32=True)


def _pad_to(t: torch.Tensor, length: int) -> torch.Tensor:
    if t.shape[-1] >= length:
        return t[..., :length]
    return torch.nn.functional.pad(t, (0, length - t.shape[-1]))
