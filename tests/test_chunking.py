"""청크 분할 + Overlap-Add 정확성 테스트.

process_fn을 항등 함수(identity)로 두면, 병합 결과는 원본과 거의 같아야 한다.
이는 Overlap-Add 가중치 정규화가 올바른지 검증한다.
"""
import torch

from src.audio.chunking import chunked_inference, make_fade_window


def test_fade_window_length():
    w = make_fade_window(100, "hann")
    assert w.shape[0] == 100


def test_input_shorter_than_chunk():
    # 짧은 클립(5초) + 긴 청크(8초): 반사 패딩 클램프가 동작해야 함
    torch.manual_seed(0)
    waveform = torch.randn(2, 44100 * 5)

    out = chunked_inference(
        waveform,
        process_fn=lambda c: c.unsqueeze(0),
        chunk_samples=44100 * 8,
        overlap=0.75,
    )
    recon = out.squeeze(0)
    assert recon.shape == waveform.shape
    assert torch.allclose(recon, waveform, atol=1e-4)


def test_identity_reconstruction():
    # 임의의 stereo 신호
    torch.manual_seed(0)
    waveform = torch.randn(2, 44100 * 3)  # 3초

    def identity(chunk: torch.Tensor) -> torch.Tensor:
        # 모델이 stem 차원을 추가한다고 가정: [1, C, T]
        return chunk.unsqueeze(0)

    out = chunked_inference(
        waveform,
        process_fn=identity,
        chunk_samples=44100,  # 1초 청크
        overlap=0.25,
        fade="hann",
    )
    # out: [1, C, T] -> stem 차원 제거
    recon = out.squeeze(0)
    assert recon.shape == waveform.shape
    # Overlap-Add 정규화가 올바르면 원본과 거의 일치
    assert torch.allclose(recon, waveform, atol=1e-4)
