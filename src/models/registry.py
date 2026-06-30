"""모델 로딩 레지스트리.

HTDemucs, BS-RoFormer, Mel-RoFormer 등 SOTA 음원 분리 모델을 통일된
인터페이스로 로드한다. 각 모델은 사전학습 가중치를 불러오거나 재학습
체크포인트를 적용할 수 있다.

TODO:
    - HTDemucs: `demucs` 패키지 활용 (get_model)
    - BS-RoFormer / Mel-RoFormer: 공식 구현 통합
    - 가중치 로딩/저장 통일 인터페이스
"""
from __future__ import annotations

import torch
import torch.nn as nn

# 지원 모델 이름
AVAILABLE_MODELS = ["htdemucs", "bs_roformer", "mel_roformer"]


def load_model(
    name: str,
    checkpoint: str | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    """이름으로 음원 분리 모델을 로드한다.

    Args:
        name: AVAILABLE_MODELS 중 하나
        checkpoint: 가중치(.pth/.ckpt) 경로. None이면 사전학습 기본 가중치.
        device: 모델을 올릴 디바이스

    Returns:
        nn.Module (입력 waveform -> stem 분리 출력)
    """
    name = name.lower()
    if name not in AVAILABLE_MODELS:
        raise ValueError(f"지원하지 않는 모델: {name} (가능: {AVAILABLE_MODELS})")

    if name == "htdemucs":
        model = _load_htdemucs(checkpoint)
    else:
        raise NotImplementedError(f"{name} 로더는 아직 구현되지 않았습니다.")

    if device is not None:
        model = model.to(device)
    return model


def _load_htdemucs(checkpoint: str | None) -> nn.Module:
    """HTDemucs 모델 로드 (demucs 패키지)."""
    try:
        from demucs.pretrained import get_model
    except ImportError as exc:
        raise ImportError(
            "demucs가 설치되지 않았습니다. `pip install demucs` 후 사용하세요."
        ) from exc

    model = get_model("htdemucs")
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state.get("state_dict", state))
    model.eval()
    return model
