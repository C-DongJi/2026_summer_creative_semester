"""디바이스 선택 유틸 테스트."""
import pytest
import torch

from src.utils.device import get_device


def test_cpu_explicit():
    assert get_device("cpu").type == "cpu"


def test_auto_returns_valid_device():
    d = get_device("auto")
    assert d.type in ("cuda", "npu", "xpu", "mps", "cpu")


def test_unavailable_backend_raises():
    # 이 테스트 환경(CPU 전용 WSL)에는 어떤 가속기도 없다
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            get_device("cuda")
    with pytest.raises(RuntimeError):
        get_device("npu")


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        get_device("tpu")
