"""재학습(Fine-tuning) 루프 (담당: 이준영).

사전학습 가중치를 불러온 뒤 커스텀 데이터로 가중치를 업데이트한다.
(설계서 Mode 2: 커스텀 재학습 파이프라인 - 학습 루프)

NOTE: 모델별 forward/출력 stem 인덱스가 다르므로 _compute_loss는
      실제 모델 통합 시 조정이 필요하다. 현재는 골격 구현.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.models.registry import load_model
from src.training.dataset import build_dataloader
from src.utils.config import Config
from src.utils.device import get_device

LOSS_FNS = {
    "l1": F.l1_loss,
    "mse": F.mse_loss,
}


def train(cfg: Config) -> None:
    """설정에 따라 Fine-tuning을 수행한다."""
    device = get_device(cfg.inference.device)
    model = load_model(
        cfg.inference.model,
        checkpoint=cfg.training.resume_from,
        device=device,
    )
    model.train()

    loader = build_dataloader(
        cfg.data.processed_dir,
        batch_size=cfg.training.batch_size,
        segment_seconds=cfg.training.segment_seconds,
        sample_rate=cfg.audio.sample_rate,
        channels=cfg.audio.channels,
        num_workers=cfg.training.num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate
    )
    loss_fn = LOSS_FNS.get(cfg.training.loss, F.l1_loss)
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.training.epochs + 1):
        running = 0.0
        pbar = tqdm(loader, desc=f"epoch {epoch}/{cfg.training.epochs}")
        for batch in pbar:
            mix = batch["mix"].to(device)
            loss = _compute_loss(model, mix, batch, loss_fn, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg = running / max(1, len(loader))
        print(f"[epoch {epoch}] avg loss = {avg:.4f}")
        _save_checkpoint(model, ckpt_dir / f"finetune_epoch{epoch}.pth", epoch)


def _compute_loss(model, mix, batch, loss_fn, device) -> torch.Tensor:
    """모델 출력과 타깃(stem) 간 손실. (모델 통합 시 stem 인덱스 조정 필요)"""
    est = model(mix)  # [B, stems, C, T] 가정
    # 예시: 첫 stem을 vocals로 본다. 실제 모델 stem 순서에 맞춰 수정할 것.
    vocals_target = batch["vocals"].to(device)
    est_vocals = est[:, 0]
    return loss_fn(est_vocals, vocals_target)


def _save_checkpoint(model, path: Path, epoch: int) -> None:
    torch.save({"epoch": epoch, "state_dict": model.state_dict()}, path)
