"""평가 지표.

SDR (Signal-to-Distortion Ratio, 글로벌/uSDR 방식):
    SDR = 10 * log10( Σ ref² / Σ (ref − est)² )

MDX 챌린지·MVSep 리더보드·MSST --metrics sdr 과 같은 계열의 정의로,
트랙 전체에 대해 한 번에 계산한다. 값이 높을수록 분리가 정확하다.
"""
from __future__ import annotations

import torch


def sdr(ref: torch.Tensor, est: torch.Tensor, eps: float = 1e-8) -> float:
    """글로벌 SDR(dB). ref/est: 동일 shape [..., T]."""
    if ref.shape != est.shape:
        n = min(ref.shape[-1], est.shape[-1])
        ref, est = ref[..., :n], est[..., :n]
    num = ref.pow(2).sum()
    den = (ref - est).pow(2).sum()
    return float(10.0 * torch.log10((num + eps) / (den + eps)))
