"""손실 함수 모듈.

Mel-Band RoFormer 체크포인트가 학습된 원 손실을 재현한다:
  waveform L1 + Σ_w L1(complex STFT(est), complex STFT(target))
  windows = [4096, 2048, 1024, 512, 256], hop = 147 (전 해상도 공통)

파인튜닝 시 체크포인트의 원 손실을 유지해야 초기 발산을 피한다
(docs/PIPELINE_DESIGN.md §3 하이퍼파라미터 참고).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    """멀티 해상도 complex STFT L1 + waveform L1."""

    def __init__(
        self,
        windows: tuple[int, ...] = (4096, 2048, 1024, 512, 256),
        hop: int = 147,
        stft_weight: float = 1.0,
    ):
        super().__init__()
        self.windows = tuple(windows)
        self.hop = hop
        self.stft_weight = stft_weight

    def forward(self, est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """est, target: [..., T] (배치/채널 차원 자유)"""
        loss = F.l1_loss(est, target)

        # STFT는 [batch, time] 2D 입력만 받으므로 앞 차원을 펼친다
        est_flat = est.reshape(-1, est.shape[-1])
        target_flat = target.reshape(-1, target.shape[-1])

        for win_len in self.windows:
            window = torch.hann_window(win_len, device=est.device, dtype=est.dtype)
            spec_est = torch.stft(
                est_flat, n_fft=win_len, hop_length=self.hop,
                window=window, return_complex=True,
            )
            spec_target = torch.stft(
                target_flat, n_fft=win_len, hop_length=self.hop,
                window=window, return_complex=True,
            )
            loss = loss + self.stft_weight * F.l1_loss(
                torch.view_as_real(spec_est), torch.view_as_real(spec_target)
            )
        return loss


def build_loss(name: str, cfg=None) -> nn.Module:
    """설정 이름으로 손실 함수를 생성한다."""
    name = name.lower()
    if name == "multi_stft":
        kwargs = {}
        if cfg is not None:
            kwargs["windows"] = tuple(cfg.training.multi_stft_windows)
            kwargs["hop"] = cfg.training.multi_stft_hop
        return MultiResolutionSTFTLoss(**kwargs)
    if name == "l1":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"알 수 없는 손실 함수: {name}")
