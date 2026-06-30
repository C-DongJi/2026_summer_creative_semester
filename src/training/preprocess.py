"""커스텀 데이터 전처리 모듈 (담당: 안준석).

운영자가 수집한 원본(Mix)과 반주(Inst) 음원 쌍을 정규화하고 노이즈를
제거하여 학습용 데이터셋으로 변환한다. (설계서 Mode 2: 재학습 파이프라인)

기대 입력 구조:
    data/raw/
        <track_name>/
            mix.wav     # 원본 (보컬 + 반주)
            inst.wav    # 반주만

출력:
    data/processed/<track_name>/{mix,inst}.wav  (정규화/리샘플 완료)
"""
from __future__ import annotations

from pathlib import Path

import torch

from src.audio.io import load_audio, save_audio


def normalize_loudness(waveform: torch.Tensor, target_peak: float = 0.95) -> torch.Tensor:
    """피크 정규화. (TODO: LUFS 기반 라우드니스 정규화로 고도화)"""
    peak = waveform.abs().max()
    if peak < 1e-8:
        return waveform
    return waveform * (target_peak / peak)


def find_pairs(raw_dir: str | Path) -> list[tuple[Path, Path]]:
    """raw 디렉토리에서 (mix, inst) 파일 쌍을 찾는다."""
    raw_dir = Path(raw_dir)
    pairs: list[tuple[Path, Path]] = []
    for track in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        mix = next(track.glob("mix.*"), None)
        inst = next(track.glob("inst.*"), None)
        if mix and inst:
            pairs.append((mix, inst))
    return pairs


def preprocess_dataset(
    raw_dir: str | Path,
    processed_dir: str | Path,
    sample_rate: int = 44100,
    channels: int = 2,
    normalize: bool = True,
) -> int:
    """raw의 모든 Mix/Inst 쌍을 전처리해 processed에 저장한다.

    Returns:
        처리한 트랙 수
    """
    processed_dir = Path(processed_dir)
    pairs = find_pairs(raw_dir)
    for mix_path, inst_path in pairs:
        track = mix_path.parent.name
        mix, _ = load_audio(mix_path, sample_rate, channels)
        inst, _ = load_audio(inst_path, sample_rate, channels)

        if normalize:
            mix = normalize_loudness(mix)
            inst = normalize_loudness(inst)

        out_dir = processed_dir / track
        save_audio(out_dir / "mix.wav", mix, sample_rate)
        save_audio(out_dir / "inst.wav", inst, sample_rate)
    return len(pairs)
