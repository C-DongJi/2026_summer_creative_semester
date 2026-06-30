"""청크 분할 추론 + Overlap-Add 모듈 (담당: 이준영).

긴 음원을 일정 길이의 청크로 분할해 추론한 뒤, 다시 병합할 때 발생하는
경계선 잡음(틱 소리 등)을 크로스페이드 윈도우로 부드럽게 연결한다.
한정된 VRAM 환경에서 OOM 없이 3~4분 음원을 처리하기 위한 핵심 로직.

핵심 아이디어:
    - chunk_samples 길이로 자르되 인접 청크가 overlap 만큼 겹치게 한다.
    - 각 청크 추론 결과에 페이드 윈도우(hann/linear)를 곱해 더한다(Overlap-Add).
    - 윈도우 가중치 합으로 나눠 정규화하면 경계가 매끄럽게 이어진다.
"""
from __future__ import annotations

from typing import Callable, Iterator

import torch


def make_fade_window(length: int, kind: str = "hann") -> torch.Tensor:
    """청크 경계 크로스페이드용 윈도우를 생성한다."""
    if kind == "hann":
        return torch.hann_window(length, periodic=False)
    if kind == "linear":
        ramp = torch.linspace(0, 1, length // 2)
        return torch.cat([ramp, ramp.flip(0)])[:length]
    raise ValueError(f"알 수 없는 윈도우 종류: {kind}")


def iter_chunks(
    waveform: torch.Tensor,
    chunk_samples: int,
    hop_samples: int,
) -> Iterator[tuple[int, torch.Tensor]]:
    """waveform[channels, samples]를 (시작 인덱스, 청크)로 순회한다.

    마지막 청크가 짧으면 0으로 패딩하여 길이를 맞춘다.
    """
    total = waveform.shape[-1]
    start = 0
    while start < total:
        end = min(start + chunk_samples, total)
        chunk = waveform[..., start:end]
        if chunk.shape[-1] < chunk_samples:
            pad = chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))
        yield start, chunk
        if end >= total:
            break
        start += hop_samples


@torch.no_grad()
def chunked_inference(
    waveform: torch.Tensor,
    process_fn: Callable[[torch.Tensor], torch.Tensor],
    chunk_samples: int,
    overlap: float = 0.25,
    fade: str = "hann",
    device: torch.device | None = None,
) -> torch.Tensor:
    """청크 분할 추론 + Overlap-Add 병합.

    Args:
        waveform: 입력 [channels, samples]
        process_fn: 청크[..., chunk_samples] -> 동일 길이 출력 추론 함수
                    (출력 stem 차원이 추가될 수 있음: [stems, channels, samples])
        chunk_samples: 청크 길이(샘플)
        overlap: 청크 간 겹침 비율 (0~1)
        fade: 크로스페이드 윈도우 종류
        device: 추론 디바이스 (None이면 입력 텐서 디바이스)

    Returns:
        병합된 출력 텐서 (process_fn 출력과 동일한 stem/channel 구조, 원본 길이로 잘림)
    """
    assert 0.0 <= overlap < 1.0, "overlap은 [0, 1) 범위여야 합니다."
    orig_total = waveform.shape[-1]
    hop_samples = max(1, int(chunk_samples * (1 - overlap)))
    window = make_fade_window(chunk_samples, fade)

    # 페이드 윈도우가 경계에서 0이 되어 음원의 맨 앞/뒤가 묻히는(weight≈0) 것을
    # 막기 위해 양 끝을 chunk_samples만큼 반사 패딩한 뒤, 마지막에 잘라낸다.
    pad = chunk_samples
    waveform = torch.nn.functional.pad(waveform, (pad, pad), mode="reflect")
    total = waveform.shape[-1]

    out_acc: torch.Tensor | None = None
    weight_acc: torch.Tensor | None = None

    for start, chunk in iter_chunks(waveform, chunk_samples, hop_samples):
        if device is not None:
            chunk = chunk.to(device)
        est = process_fn(chunk)            # [..., chunk_samples]
        est = est.cpu()

        # 출력 구조에 맞춰 누적 버퍼 초기화 (지연 초기화)
        if out_acc is None:
            tail_shape = est.shape[:-1]                      # stem/channel 차원
            out_acc = torch.zeros(*tail_shape, total)
            weight_acc = torch.zeros(total)

        win = window.to(est.dtype)
        seg_len = min(chunk_samples, total - start)
        out_acc[..., start:start + seg_len] += (est * win)[..., :seg_len]
        weight_acc[start:start + seg_len] += win[:seg_len]

    # 0으로 나누기 방지 후 정규화, 패딩 영역 제거
    weight_acc = weight_acc.clamp_min(1e-8)
    out = out_acc / weight_acc
    return out[..., pad:pad + orig_total]
