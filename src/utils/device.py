"""디바이스 선택 유틸.

지원 가속기: cuda(NVIDIA) / npu(화웨이 Ascend, torch_npu 플러그인 필요) /
xpu(Intel GPU) / mps(Apple Silicon) / cpu

NPU 주의사항:
  - PyTorch에서 'npu' 디바이스는 torch_npu 패키지를 설치해야 활성화된다.
  - Mel-Band RoFormer는 내부에서 torch.stft/복소 연산을 사용하는데, NPU/XPU
    백엔드는 이들 연산 지원이 제한적일 수 있다. 추론이 연산 미지원 오류로
    실패하면 config의 inference.device를 cpu(또는 cuda)로 되돌릴 것.
"""
from __future__ import annotations

import torch


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401  (설치 시 torch.npu 네임스페이스 활성화)
    except ImportError:
        pass
    npu = getattr(torch, "npu", None)
    return npu is not None and npu.is_available()


def _xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    return xpu is not None and xpu.is_available()


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_available()


def get_device(prefer: str = "auto") -> torch.device:
    """사용할 torch 디바이스를 결정한다.

    Args:
        prefer: "auto" | "cuda" | "npu" | "xpu" | "mps" | "cpu"
                auto는 cuda > npu > xpu > mps > cpu 순으로 선택.
    """
    prefer = (prefer or "auto").lower()

    if prefer == "cpu":
        return torch.device("cpu")

    checks = {
        "cuda": torch.cuda.is_available,
        "npu": _npu_available,
        "xpu": _xpu_available,
        "mps": _mps_available,
    }

    if prefer in checks:
        if not checks[prefer]():
            hint = " (torch_npu 플러그인 설치 필요)" if prefer == "npu" else ""
            raise RuntimeError(f"{prefer}를 사용할 수 없습니다{hint}.")
        return torch.device(prefer)

    if prefer != "auto":
        raise ValueError(f"알 수 없는 device 설정: {prefer}")

    for name, available in checks.items():
        if available():
            return torch.device(name)
    return torch.device("cpu")
