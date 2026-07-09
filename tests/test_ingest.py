"""데이터셋 인제스트 공통 유틸 테스트."""
import torch

from src.data.ingest_common import pad_and_sum, to_stereo_tensor


def test_to_stereo_from_mono():
    mono = torch.randn(1, 100)
    out = to_stereo_tensor(mono, 2)
    assert out.shape == (2, 100)
    assert torch.equal(out[0], out[1])  # mono는 좌우 동일


def test_to_stereo_from_1d():
    out = to_stereo_tensor(torch.randn(50), 2)
    assert out.shape == (2, 50)


def test_to_stereo_downmix():
    out = to_stereo_tensor(torch.randn(4, 30), 2)
    assert out.shape == (2, 30)  # 초과 채널은 앞쪽 취함


def test_pad_and_sum_unequal_lengths():
    a = torch.ones(2, 10)
    b = torch.ones(2, 15)
    out = pad_and_sum([a, b])
    assert out.shape == (2, 15)
    assert out[0, 0].item() == 2.0   # 두 신호 겹치는 구간
    assert out[0, 12].item() == 1.0  # b만 있는 구간


def test_pad_and_sum_empty():
    assert pad_and_sum([]) is None
    assert pad_and_sum([None, None]) is None
