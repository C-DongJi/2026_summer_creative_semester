"""디바이스 선택 유틸."""
from __future__ import annotations

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """사용할 torch 디바이스를 결정한다.

    Args:
        prefer: "auto" | "cuda" | "cpu"
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없습니다.")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
