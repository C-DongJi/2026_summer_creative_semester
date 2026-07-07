"""재학습용 PyTorch Dataset / DataLoader (담당: 안준석).

MSST dataset_type 4 레이아웃(<곡명>/{vocals.wav, other.wav})을 읽어
일정 길이 세그먼트로 잘라 학습에 공급한다.

mixture는 스템 합(vocals + other)으로 재구성한다 — 전처리에서
vocals = mix − inst 로 유도했으므로 합치면 원본 mix와 일치한다.
(MSST 데이터 로더의 mixture 처리 방식과 동일)
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.audio.io import load_audio


class VocalSeparationDataset(Dataset):
    """(vocals, other) 스템 쌍에서 랜덤 세그먼트를 추출하는 데이터셋.

    각 샘플: {"mixture": [C, T], "vocals": [C, T], "other": [C, T]}
    """

    def __init__(
        self,
        processed_dir: str | Path,
        segment_seconds: float = 3.0,
        sample_rate: int = 44100,
        channels: int = 2,
    ):
        self.processed_dir = Path(processed_dir)
        self.sample_rate = sample_rate
        self.channels = channels
        self.segment_samples = int(segment_seconds * sample_rate)
        self.tracks = sorted(
            p for p in self.processed_dir.iterdir()
            if p.is_dir() and (p / "vocals.wav").exists() and (p / "other.wav").exists()
        )
        if not self.tracks:
            raise RuntimeError(
                f"전처리된 트랙이 없습니다: {self.processed_dir} "
                "(scripts/prepare_data.py 를 먼저 실행하세요)"
            )

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        track = self.tracks[idx]
        vocals, _ = load_audio(track / "vocals.wav", self.sample_rate, self.channels)
        other, _ = load_audio(track / "other.wav", self.sample_rate, self.channels)

        length = min(vocals.shape[-1], other.shape[-1])
        vocals, other = vocals[..., :length], other[..., :length]

        vocals, other = self._random_segment(vocals, other)
        return {"mixture": vocals + other, "vocals": vocals, "other": other}

    def _random_segment(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """동일한 랜덤 위치에서 segment_samples 길이를 잘라낸다 (부족하면 패딩)."""
        total = a.shape[-1]
        seg = self.segment_samples
        if total <= seg:
            pad = seg - total
            return (
                torch.nn.functional.pad(a, (0, pad)),
                torch.nn.functional.pad(b, (0, pad)),
            )
        start = int(torch.randint(0, total - seg + 1, (1,)).item())
        return a[..., start:start + seg], b[..., start:start + seg]


def build_dataloader(
    processed_dir: str | Path,
    batch_size: int = 1,
    segment_seconds: float = 3.0,
    sample_rate: int = 44100,
    channels: int = 2,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """학습용 DataLoader를 생성한다."""
    dataset = VocalSeparationDataset(
        processed_dir, segment_seconds, sample_rate, channels
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
