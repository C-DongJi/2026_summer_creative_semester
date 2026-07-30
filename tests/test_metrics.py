"""SDR 지표 및 평가 스크립트 헬퍼 테스트."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate import parse_labeled  # noqa: E402
from src.utils.metrics import sdr  # noqa: E402


def test_sdr_perfect_estimate_is_high():
    x = torch.randn(2, 44100)
    assert sdr(x, x) > 80.0  # 오차 0이면 eps에 의해 매우 큰 값


def test_sdr_known_ratio():
    # ref 에너지 1, 오차 에너지 0.01 -> 20 dB
    ref = torch.ones(1, 1000)
    est = ref + 0.1
    assert sdr(ref, est) == pytest.approx(20.0, abs=0.01)


def test_sdr_worse_estimate_is_lower():
    torch.manual_seed(0)
    ref = torch.randn(2, 8000)
    near = ref + 0.01 * torch.randn_like(ref)
    far = ref + 0.5 * torch.randn_like(ref)
    assert sdr(ref, near) > sdr(ref, far)


def test_sdr_length_mismatch_trimmed():
    ref = torch.ones(1, 1000)
    est = torch.ones(1, 1200)
    assert sdr(ref, est) > 80.0  # 공통 구간만 비교


def test_parse_labeled():
    assert parse_labeled(["a=1", "b=x/y"], "--m") == {"a": "1", "b": "x/y"}
    assert parse_labeled([], "--m") == {}
    with pytest.raises(SystemExit):
        parse_labeled(["noequals"], "--m")
    with pytest.raises(SystemExit):
        parse_labeled(["a=1", "a=2"], "--m")  # 이름 중복
