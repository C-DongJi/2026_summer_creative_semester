"""MedleyDB → 2-stem(보컬/반주) 변환.

MedleyDB(V1 122곡 / 2.0까지 196곡, 44.1kHz, 스템은 스테레오, CC BY-NC-SA)를
공식 패키지로 로드해 보컬/반주 쌍을 생성한다.

설치:  pip install medleydb
환경:  export MEDLEYDB_PATH=/path/to/MedleyDB   (하위에 Audio/ 폴더)
로드:  medleydb.load_all_multitracks(dataset_version=['V1','V2']) -> MultiTrack 제너레이터
      mtrack.stems         -> {int: Track}
      mtrack.is_instrumental / mtrack.has_bleed (bool)
      track.instrument     -> list[str]  (항상 리스트)
      track.audio_path     -> 스템 wav 경로 (MEDLEYDB_PATH 설정 시)

주의:
  - 트랙의 약 40~45%가 순수 반주(instrumental) — is_instrumental로 건너뜀.
  - has_bleed(스템 간 누화) 트랙은 깨끗한 타깃을 위해 기본 제외.
  - mixture는 마스터링된 _MIX.wav가 아니라 '스템 합'으로 정의(자기정합적).
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch

from src.audio.io import load_audio
from src.data.ingest_common import (
    VOICE_LABELS_SINGING,
    VOICE_LABELS_SPEECH,
    pad_and_sum,
)


def iter_medleydb_pairs(
    versions: list[str] | None = None,
    include_speech: bool = False,
    drop_bleed: bool = True,
    sample_rate: int = 44100,
    channels: int = 2,
) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
    """MedleyDB의 각 트랙에서 (트랙명, vocals[C,T], other[C,T])를 순회.

    Args:
        versions: ['V1'], ['V1','V2'] 등. None이면 ['V1','V2'].
        include_speech: True면 speaker/crowd도 보컬로 취급.
        drop_bleed: True면 has_bleed 트랙 제외.
    """
    try:
        import medleydb
    except ImportError as exc:
        raise ImportError(
            "medleydb 패키지가 없습니다.\n"
            "  pip install medleydb\n"
            "그리고 `export MEDLEYDB_PATH=/path/to/MedleyDB` (하위에 Audio/) 를 설정하세요."
        ) from exc

    vocal_labels = set(VOICE_LABELS_SINGING)
    if include_speech:
        vocal_labels |= VOICE_LABELS_SPEECH

    versions = versions or ["V1", "V2"]
    for mtrack in medleydb.load_all_multitracks(dataset_version=versions):
        if getattr(mtrack, "is_instrumental", False):
            continue
        if drop_bleed and getattr(mtrack, "has_bleed", False):
            continue

        vocal_stems: list[torch.Tensor] = []
        other_stems: list[torch.Tensor] = []
        for stem in mtrack.stems.values():
            path = getattr(stem, "audio_path", None)
            if not path or not Path(path).exists():
                # 오디오 미설치(MEDLEYDB_PATH 미설정 등) — 스킵 대신 명확히 알림
                raise FileNotFoundError(
                    f"스템 오디오를 찾을 수 없습니다: {path}\n"
                    "MEDLEYDB_PATH가 올바른지, Audio/가 존재하는지 확인하세요."
                )
            wav, _ = load_audio(path, sample_rate, channels)
            labels = stem.instrument if isinstance(stem.instrument, list) else [stem.instrument]
            if any(lbl in vocal_labels for lbl in labels):
                vocal_stems.append(wav)
            else:
                other_stems.append(wav)

        vocals = pad_and_sum(vocal_stems)
        other = pad_and_sum(other_stems)
        if vocals is None or other is None:
            # 보컬 스템이 하나도 없거나(라벨상 비보컬) 반주가 없으면 부적합
            continue

        name = f"medleydb__{mtrack.track_id}"
        yield name, vocals, other
