"""MoisesDB → 2-stem(보컬/반주) 변환.

MoisesDB(240곡, 11개 top-level 스템, 44.1kHz 스테레오, CC BY-NC-SA 4.0)를
공식 패키지로 로드해 보컬/반주 쌍을 생성한다.

설치:  pip install git+https://github.com/moises-ai/moises-db.git
로드:  MoisesDB(data_path=..., sample_rate=44100), db[i] -> MoisesDBTrack
      track.stems  -> {stem_name: ndarray[C, T]}  (보컬 키 = "vocals")
      track.audio  -> 전체 스템의 합 = mixture
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch

from src.data.ingest_common import pad_and_sum, to_stereo_tensor


def iter_moisesdb_pairs(
    data_path: str | Path,
    sample_rate: int = 44100,
    channels: int = 2,
) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
    """MoisesDB의 각 트랙에서 (트랙명, vocals[C,T], other[C,T])를 순회.

    보컬이 없는 트랙은 건너뛴다.
    """
    try:
        from moisesdb.dataset import MoisesDB
    except ImportError as exc:
        raise ImportError(
            "moisesdb 패키지가 없습니다.\n"
            "  pip install git+https://github.com/moises-ai/moises-db.git\n"
            "그리고 데이터셋 경로(MOISESDB_PATH 또는 --src)를 지정하세요."
        ) from exc

    db = MoisesDB(data_path=str(data_path), sample_rate=sample_rate)
    for track in db:
        # 보컬 없는 트랙은 오디오 로드 전에 건너뜀 (metadata 기반)
        sources = getattr(track, "sources", None)
        if sources is not None and "vocals" not in sources:
            continue

        stems = track.stems  # {name: ndarray[C, T]} — 보컬 있는 트랙만 여기 진입
        if "vocals" not in stems:
            continue

        vocals = to_stereo_tensor(stems["vocals"], channels)
        other = pad_and_sum(
            [to_stereo_tensor(v, channels) for k, v in stems.items() if k != "vocals"]
        )
        if other is None:  # 보컬만 있고 반주가 없는 트랙은 학습에 부적합
            continue

        name = _track_name(track)
        yield name, vocals, other


def _track_name(track) -> str:
    """파일시스템에 안전한 고유 트랙명 생성."""
    provider = getattr(track, "provider", "x")
    tid = getattr(track, "id", None) or getattr(track, "song", "unknown")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"{provider}_{tid}")
    return f"moisesdb__{safe}"
