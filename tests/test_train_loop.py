"""자체 학습 루프(src/training/train.py) dry-run 테스트.

실제 RoFormer 대신 초소형 더미 모델을 주입해 루프 전체
(DataLoader -> forward -> loss -> grad accumulation -> checkpoint 저장)가
CPU에서 끝까지 도는지 검증한다.
"""
import torch
import torch.nn as nn

import src.training.train as train_mod
from src.data.ingest_common import save_stem_pair
from src.utils.config import Config

SR = 44100


class TinyModel(nn.Module):
    """[B, C, T] -> [B, C, T] 형태만 흉내내는 더미 (단일 타깃 모델 규격)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 2, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


def _make_cfg(tmp_path, loss_name: str) -> Config:
    return Config({
        "audio": {"sample_rate": SR, "channels": 2},
        "inference": {"device": "cpu"},
        "model": {"name": "mel_band_roformer"},  # monkeypatch로 대체됨
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 2,
            "learning_rate": 1e-3,
            "num_workers": 0,
            "segment_seconds": 0.5,
            "use_amp": False,
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "loss": loss_name,
            "multi_stft_windows": [1024, 512],  # 테스트 속도용 축소
            "multi_stft_hop": 147,
        },
        "data": {"processed_dir": str(tmp_path / "proc")},
    })


def _make_dataset(tmp_path, n_tracks: int = 2) -> None:
    torch.manual_seed(0)
    for i in range(n_tracks):
        save_stem_pair(
            tmp_path / "proc" / f"track{i}",
            torch.randn(2, SR) * 0.1,
            torch.randn(2, SR) * 0.1,
            SR,
        )


def test_train_loop_completes_and_saves_checkpoint(tmp_path, monkeypatch):
    _make_dataset(tmp_path)
    cfg = _make_cfg(tmp_path, loss_name="l1")
    monkeypatch.setattr(train_mod, "load_model", lambda m, device=None: TinyModel())

    train_mod.train(cfg)

    ckpt = tmp_path / "ckpt" / "finetune_epoch1.ckpt"
    assert ckpt.exists()
    state = torch.load(ckpt, map_location="cpu")
    assert state["epoch"] == 1
    assert "conv.weight" in state["state_dict"]


def test_train_loop_with_multi_stft_loss(tmp_path, monkeypatch):
    _make_dataset(tmp_path)
    cfg = _make_cfg(tmp_path, loss_name="multi_stft")
    monkeypatch.setattr(train_mod, "load_model", lambda m, device=None: TinyModel())

    train_mod.train(cfg)  # 예외 없이 완주하면 통과
    assert (tmp_path / "ckpt" / "finetune_epoch1.ckpt").exists()


def test_train_loop_updates_weights(tmp_path, monkeypatch):
    _make_dataset(tmp_path)
    cfg = _make_cfg(tmp_path, loss_name="l1")
    model = TinyModel()
    before = model.conv.weight.detach().clone()
    monkeypatch.setattr(train_mod, "load_model", lambda m, device=None: model)

    train_mod.train(cfg)

    assert not torch.equal(before, model.conv.weight.detach()), "가중치가 업데이트되지 않음"
