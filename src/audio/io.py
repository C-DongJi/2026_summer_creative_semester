"""오디오 입출력 모듈 (담당: 안준석).

다양한 포맷(MP3/WAV/FLAC 등)과 샘플레이트의 오디오를 모델 입력 규격
(목표 샘플레이트, 채널 수)에 맞게 로드/변환/저장한다.

TODO:
    - MP3 디코딩 (torchaudio 백엔드 또는 ffmpeg)
    - 샘플레이트 리샘플링
    - mono/stereo 채널 변환
    - 라우드니스 정규화
"""
from __future__ import annotations

from pathlib import Path

import torch
import torchaudio


def load_audio(
    path: str | Path,
    target_sr: int = 44100,
    target_channels: int = 2,
) -> tuple[torch.Tensor, int]:
    """오디오 파일을 로드하여 모델 입력 규격으로 변환한다.

    Args:
        path: 입력 오디오 경로 (mp3/wav/flac/...)
        target_sr: 목표 샘플레이트
        target_channels: 목표 채널 수 (1=mono, 2=stereo)

    Returns:
        (waveform[channels, samples], sample_rate)
    """
    waveform, sr = torchaudio.load(str(path))

    # 샘플레이트 변환
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    # 채널 변환
    waveform = _match_channels(waveform, target_channels)
    return waveform, sr


def save_audio(
    path: str | Path,
    waveform: torch.Tensor,
    sample_rate: int,
) -> None:
    """waveform[channels, samples]를 파일로 저장한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform.cpu(), sample_rate)


def _match_channels(waveform: torch.Tensor, target_channels: int) -> torch.Tensor:
    """채널 수를 목표값에 맞춘다."""
    cur = waveform.shape[0]
    if cur == target_channels:
        return waveform
    if cur == 1 and target_channels == 2:
        return waveform.repeat(2, 1)
    if cur == 2 and target_channels == 1:
        return waveform.mean(dim=0, keepdim=True)
    # 그 외: 앞쪽 채널을 취하거나 평균
    if cur > target_channels:
        return waveform[:target_channels]
    return waveform.repeat(target_channels, 1)[:target_channels]
