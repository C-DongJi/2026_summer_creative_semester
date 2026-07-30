"""재학습용 PyTorch Dataset / DataLoader (담당: 안준석).

[창의학기제 5주차 (7/20) 산출물 — Dataset/DataLoader 모듈 설계]

MSST dataset_type 4 레이아웃(<곡명>/{vocals.wav, other.wav})을 읽어
일정 길이 세그먼트로 잘라 학습에 공급한다.

mixture는 스템 합(vocals + other)으로 재구성한다 — 전처리에서
vocals = mix − inst 로 유도했으므로 합치면 원본 mix와 일치한다.

I/O 최적화: 트랙 전체를 디코드하지 않고 soundfile 부분 읽기로
필요한 세그먼트 구간만 읽는다 (4분 트랙 기준 아이템당 ~169MB → ~2MB).
전처리 단계가 목표 샘플레이트/채널로 저장해 두므로 여기서는 변환하지 않고
__init__에서 규격만 1회 검증한다.
"""
from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset


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

        tracks = sorted(
            p for p in self.processed_dir.iterdir()
            if p.is_dir() and (p / "vocals.wav").exists() and (p / "other.wav").exists()
        )
        if not tracks:
            raise RuntimeError(
                f"전처리된 트랙이 없습니다: {self.processed_dir} "
                "(scripts/prepare_data.py 를 먼저 실행하세요)"
            )

        # 트랙별 프레임 수 캐시 + 규격 1회 검증 (이후 부분 읽기만 수행)
        self.tracks: list[Path] = []
        self.frames: list[int] = []
        for track in tracks:
            v_info = sf.info(str(track / "vocals.wav"))
            o_info = sf.info(str(track / "other.wav"))
            if v_info.samplerate != sample_rate or o_info.samplerate != sample_rate:
                raise ValueError(
                    f"{track.name}: 샘플레이트 {v_info.samplerate}/{o_info.samplerate} "
                    f"(기대 {sample_rate}) — prepare_data.py로 재전처리하세요."
                )
            if v_info.channels != channels or o_info.channels != channels:
                raise ValueError(
                    f"{track.name}: 채널 {v_info.channels}/{o_info.channels} "
                    f"(기대 {channels}) — prepare_data.py로 재전처리하세요."
                )
            self.tracks.append(track)
            self.frames.append(min(v_info.frames, o_info.frames))

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        track = self.tracks[idx]
        total = self.frames[idx]
        seg = self.segment_samples

        if total <= seg:
            start, frames = 0, total
        else:
            start = int(torch.randint(0, total - seg + 1, (1,)).item())
            frames = seg

        vocals = self._read_window(track / "vocals.wav", start, frames)
        other = self._read_window(track / "other.wav", start, frames)

        if frames < seg:  # 짧은 트랙은 zero-pad
            pad = seg - frames
            vocals = torch.nn.functional.pad(vocals, (0, pad))
            other = torch.nn.functional.pad(other, (0, pad))

        return {"mixture": vocals + other, "vocals": vocals, "other": other}

    @staticmethod
    def _read_window(path: Path, start: int, frames: int) -> torch.Tensor:
        """[start, start+frames) 구간만 디코드해 [C, frames] 텐서로 반환."""
        data, _sr = sf.read(
            str(path), start=start, frames=frames,
            dtype="float32", always_2d=True,
        )  # (frames, channels)
        return torch.from_numpy(data.T).contiguous()


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
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
