"""자체 재학습(Fine-tuning) 루프 (담당: 이준영).

[창의학기제 6주차 (7/27) 산출물 — Fine-tuning 루프 설계 및 환경 모듈화]

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

    # training.resume_from이 지정되면 그 체크포인트에서 재개 (기본은 model.checkpoint)
    model_cfg = cfg.model
    resume_from = getattr(cfg.training, "resume_from", None)
    if resume_from:
        merged = dict(model_cfg)
        merged["checkpoint"] = resume_from
        model_cfg = Config(merged)
    model = load_model(model_cfg, device=device)
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

    total_steps = len(loader)
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
            # 에폭 마지막 배치에서는 축적이 덜 찼어도 반드시 스텝 —
            # 그러지 않으면 남은 그래디언트가 다음 에폭의 zero_grad로 버려지고,
            # 트랙 수 < accum이면 옵티마이저가 영영 돌지 않는다.
            if step % accum == 0 or step == total_steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running += loss.item() * accum
            pbar.set_postfix(loss=f"{loss.item() * accum:.4f}")

        avg = running / max(1, total_steps)
        print(f"[epoch {epoch}] avg loss = {avg:.4f}")
        _save_checkpoint(model, ckpt_dir, epoch, keep_last=3)


def _forward_vocals(model: torch.nn.Module, mixture: torch.Tensor) -> torch.Tensor:
    """모델 출력을 vocals 추정 [B, C, T]로 정규화한다."""
    est = model(mixture)
    if est.dim() == 4:  # [B, S, C, T] 멀티 스템 출력 — vocals 스템 선택
        sources = getattr(model, "sources", None)
        idx = sources.index("vocals") if sources and "vocals" in sources else 0
        est = est[:, idx]
    return est


def _save_checkpoint(
    model: torch.nn.Module, ckpt_dir: Path, epoch: int, keep_last: int = 3
) -> None:
    """에폭 체크포인트 저장 후 최근 keep_last개만 보관 (228M 모델 = 개당 ~0.9GB)."""
    path = ckpt_dir / f"finetune_epoch{epoch}.ckpt"
    torch.save({"epoch": epoch, "state_dict": model.state_dict()}, path)
    old = sorted(
        ckpt_dir.glob("finetune_epoch*.ckpt"),
        key=lambda p: int(p.stem.replace("finetune_epoch", "")),
    )
    for stale in old[:-keep_last]:
        stale.unlink()
