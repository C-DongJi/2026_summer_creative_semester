"""자체 재학습(Fine-tuning) 루프 (담당: 이준영).

창의학기제 학습 목표(학습 루프 직접 설계·구현)를 위한 경량 구현.
검증된 기본 경로는 MSST train.py 사용 — docs/PIPELINE_DESIGN.md Mode 2 참고.

구성: 단일 타깃(vocals) 학습
  - 모델 입력: mixture [B, C, T] -> 출력: vocals 추정 [B, C, T]
  - 손실: multi-resolution STFT (체크포인트 원 손실 유지)
  - VRAM 절감: AMP(mixed precision) + gradient accumulation
"""
from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from src.models.registry import load_model
from src.training.dataset import build_dataloader
from src.training.losses import build_loss
from src.utils.config import Config
from src.utils.device import get_device


def train(cfg: Config) -> None:
    """설정에 따라 Fine-tuning을 수행한다."""
    device = get_device(cfg.inference.device)
    model = load_model(cfg.model, device=device)
    model.train()

    loader = build_dataloader(
        cfg.data.processed_dir,
        batch_size=cfg.training.batch_size,
        segment_seconds=cfg.training.segment_seconds,
        sample_rate=cfg.audio.sample_rate,
        channels=cfg.audio.channels,
        num_workers=cfg.training.num_workers,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    loss_fn = build_loss(cfg.training.loss, cfg).to(device)
    accum = max(1, cfg.training.gradient_accumulation_steps)
    use_amp = bool(cfg.training.use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.training.epochs + 1):
        running = 0.0
        optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"epoch {epoch}/{cfg.training.epochs}")
        for step, batch in enumerate(pbar, start=1):
            mixture = batch["mixture"].to(device)
            vocals = batch["vocals"].to(device)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                est = _forward_vocals(model, mixture)
                loss = loss_fn(est, vocals) / accum

            scaler.scale(loss).backward()
            if step % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running += loss.item() * accum
            pbar.set_postfix(loss=f"{loss.item() * accum:.4f}")

        avg = running / max(1, len(loader))
        print(f"[epoch {epoch}] avg loss = {avg:.4f}")
        _save_checkpoint(model, ckpt_dir / f"finetune_epoch{epoch}.ckpt", epoch)


def _forward_vocals(model: torch.nn.Module, mixture: torch.Tensor) -> torch.Tensor:
    """모델 출력을 vocals 추정 [B, C, T]로 정규화한다."""
    est = model(mixture)
    if est.dim() == 4:  # [B, S, C, T] 멀티 스템 출력 — vocals 스템 선택
        sources = getattr(model, "sources", None)
        idx = sources.index("vocals") if sources and "vocals" in sources else 0
        est = est[:, idx]
    return est


def _save_checkpoint(model: torch.nn.Module, path: Path, epoch: int) -> None:
    torch.save({"epoch": epoch, "state_dict": model.state_dict()}, path)
