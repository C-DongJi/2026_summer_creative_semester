"""재학습용 PyTorch Dataset / DataLoader (담당: 안준석).

전처리된 Mix/Inst 쌍을 일정 길이 세그먼트로 잘라 학습에 공급한다.
보컬 타깃은 vocal = mix - inst 로 유도한다.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.audio.io import load_audio


class StemSeparationDataset(Dataset):
    """(mix, inst) 쌍에서 랜덤 세그먼트를 추출하는 데이터셋.

    각 샘플: {"mix": [C, T], "inst": [C, T], "vocals": [C, T]}
    """

    def __init__(
        self,
        processed_dir: str | Path,
        segment_seconds: float = 6.0,
        sample_rate: int = 44100,
        channels: int = 2,
    ):
        self.processed_dir = Path(processed_dir)
        self.sample_rate = sample_rate
        self.channels = channels
        self.segment_samples = int(segment_seconds * sample_rate)
        self.tracks = sorted(
            p for p in self.processed_dir.iterdir()
            if p.is_dir() and (p / "mix.wav").exists()
        )
        if not self.tracks:
            raise RuntimeError(f"전처리된 트랙이 없습니다: {self.processed_dir}")

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        track = self.tracks[idx]
        mix, _ = load_audio(track / "mix.wav", self.sample_rate, self.channels)
        inst, _ = load_audio(track / "inst.wav", self.sample_rate, self.channels)

        # 길이 맞춤
        length = min(mix.shape[-1], inst.shape[-1])
        mix, inst = mix[..., :length], inst[..., :length]

        mix, inst = self._random_segment(mix, inst)
        vocals = mix - inst
        return {"mix": mix, "inst": inst, "vocals": vocals}

    def _random_segment(
        self, mix: torch.Tensor, inst: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """랜덤 위치에서 segment_samples 길이를 잘라낸다 (부족하면 패딩)."""
        total = mix.shape[-1]
        seg = self.segment_samples
        if total <= seg:
            pad = seg - total
            mix = torch.nn.functional.pad(mix, (0, pad))
            inst = torch.nn.functional.pad(inst, (0, pad))
            return mix, inst
        start = int(torch.randint(0, total - seg + 1, (1,)).item())
        return mix[..., start:start + seg], inst[..., start:start + seg]


def build_dataloader(
    processed_dir: str | Path,
    batch_size: int = 4,
    segment_seconds: float = 6.0,
    sample_rate: int = 44100,
    channels: int = 2,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """학습용 DataLoader를 생성한다."""
    dataset = StemSeparationDataset(
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
