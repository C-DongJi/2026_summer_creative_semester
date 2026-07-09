"""오디오 입출력 모듈 (담당: 안준석).

다양한 포맷(MP3/WAV/FLAC 등)과 샘플레이트의 오디오를 모델 입력 규격
(목표 샘플레이트, 채널 수)에 맞게 로드/변환/저장한다.

I/O 백엔드는 soundfile(libsndfile)을 사용한다. torchaudio 2.9+는 load/save가
TorchCodec 백엔드를 요구해 환경 의존성이 커지므로, 견고한 soundfile로 읽고 쓰고
리샘플링만 torchaudio.functional을 쓴다(백엔드 불필요).
"""
from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF


def load_audio(
    path: str | Path,
    target_sr: int = 44100,
    target_channels: int = 2,
) -> tuple[torch.Tensor, int]:
    """오디오 파일을 로드하여 모델 입력 규격으로 변환한다.

    Args:
        path: 입력 오디오 경로 (wav/flac/ogg/mp3 등, libsndfile 지원 포맷)
        target_sr: 목표 샘플레이트
        target_channels: 목표 채널 수 (1=mono, 2=stereo)

    Returns:
        (waveform[channels, samples], sample_rate)
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (frames, channels)
    waveform = torch.from_numpy(data.T).contiguous()  # (channels, frames)

    if sr != target_sr:
        waveform = AF.resample(waveform, sr, target_sr)
        sr = target_sr

    waveform = _match_channels(waveform, target_channels)
    return waveform, sr


def save_audio(
    path: str | Path,
    waveform: torch.Tensor,
    sample_rate: int,
    float32: bool = False,
) -> None:
    """waveform[channels, samples]를 파일로 저장한다.

    Args:
        float32: True면 32-bit float WAV로 저장 (감산/합산으로 유도한 스템 등
                 양자화 손실을 피해야 하는 학습 데이터용)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = waveform.detach().cpu().numpy().T  # (frames, channels)
    subtype = "FLOAT" if float32 else "PCM_16"
    sf.write(str(path), data, sample_rate, subtype=subtype)


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
