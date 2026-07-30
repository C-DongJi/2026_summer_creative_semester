"""청크 분할 + Overlap-Add 정확성 테스트.

process_fn을 항등 함수(identity)로 두면, 병합 결과는 원본과 거의 같아야 한다.
이는 Overlap-Add 가중치 정규화가 올바른지 검증한다.
process_fn은 배치 인터페이스: [B, C, T] -> [B, (S,) C, T]
"""
import pytest
import torch

from src.audio.chunking import chunked_inference, make_fade_window


def test_fade_window_length():
    assert make_fade_window(100, "hann").shape[0] == 100
    assert make_fade_window(100, "linear").shape[0] == 100
    # 홀수 길이에서도 정확한 길이 보장 (과거 linear 윈도우 크래시 회귀 테스트)
    assert make_fade_window(501, "hann").shape[0] == 501
    assert make_fade_window(501, "linear").shape[0] == 501


def test_input_shorter_than_chunk():
    # 짧은 클립(5초) + 긴 청크(8초): 반사 패딩 클램프가 동작해야 함
    torch.manual_seed(0)
    waveform = torch.randn(2, 44100 * 5)

    out = chunked_inference(
        waveform,
        process_fn=lambda b: b,  # [B, C, T] 항등
        chunk_samples=44100 * 8,
        overlap=0.75,
    )
    assert out.shape == waveform.shape
    assert torch.allclose(out, waveform, atol=1e-4)


def test_identity_reconstruction():
    torch.manual_seed(0)
    waveform = torch.randn(2, 44100 * 3)  # 3초 stereo

    out = chunked_inference(
        waveform,
        process_fn=lambda b: b,
        chunk_samples=44100,  # 1초 청크
        overlap=0.25,
        fade="hann",
    )
    assert out.shape == waveform.shape
    assert torch.allclose(out, waveform, atol=1e-4)


def test_identity_with_stem_dim():
    # 모델이 stem 차원을 추가하는 경우: [B, C, T] -> [B, 1, C, T]
    torch.manual_seed(0)
    waveform = torch.randn(2, 44100 * 2)

    out = chunked_inference(
        waveform,
        process_fn=lambda b: b.unsqueeze(1),
        chunk_samples=44100,
        overlap=0.5,
    )
    assert out.shape == (1, 2, waveform.shape[-1])
    assert torch.allclose(out.squeeze(0), waveform, atol=1e-4)


def test_batched_equals_unbatched():
    # 청크 배칭(batch_size>1)이 결과를 바꾸지 않아야 함
    torch.manual_seed(1)
    waveform = torch.randn(2, 44100 * 3)

    kwargs = dict(process_fn=lambda b: b, chunk_samples=44100, overlap=0.75)
    out1 = chunked_inference(waveform, batch_size=1, **kwargs)
    out4 = chunked_inference(waveform, batch_size=4, **kwargs)
    assert torch.allclose(out1, out4, atol=1e-6)


def test_odd_chunk_with_linear_fade():
    # 홀수 청크 길이 + linear 페이드 조합 회귀 테스트
    torch.manual_seed(2)
    waveform = torch.randn(2, 2000)
    out = chunked_inference(
        waveform, process_fn=lambda b: b, chunk_samples=501,
        overlap=0.75, fade="linear",
    )
    assert out.shape == waveform.shape
    assert torch.allclose(out, waveform, atol=1e-3)


def test_zero_overlap_rejected():
    # overlap=0은 경계 드롭아웃을 만들므로 거부되어야 함
    with pytest.raises(AssertionError):
        chunked_inference(
            torch.randn(2, 44100), process_fn=lambda b: b,
            chunk_samples=22050, overlap=0.0,
        )
