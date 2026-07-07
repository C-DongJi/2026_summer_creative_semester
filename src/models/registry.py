"""모델 로딩 레지스트리 (2026-07 재설계).

주력: Mel-Band RoFormer (Kim 체크포인트, MIT) — ZFTurbo MSST 프레임워크의
모델 구현을 통해 로드한다. MSST는 pip 패키지가 아니므로 리포 체크아웃
(scripts/setup_msst.sh)을 sys.path에 추가해 사용한다.

HTDemucs는 레거시 베이스라인 비교용으로만 유지한다
(리포 아카이브됨 + 4-stem 고정이라 (mix, inst) 쌍 파인튜닝 불가).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

AVAILABLE_MODELS = ["mel_band_roformer", "bs_roformer", "htdemucs"]


def load_model(model_cfg, device: torch.device | None = None) -> nn.Module:
    """설정(config의 model 섹션)으로 음원 분리 모델을 로드한다.

    Args:
        model_cfg: name / msst_dir / msst_config / checkpoint 필드를 가진 설정
        device: 모델을 올릴 디바이스

    Returns:
        nn.Module — 단일 타깃 모델은 [B, C, T] -> [B, C, T](vocals),
        멀티 스템 모델은 [B, C, T] -> [B, S, C, T]
    """
    name = model_cfg.name.lower()
    if name not in AVAILABLE_MODELS:
        raise ValueError(f"지원하지 않는 모델: {name} (가능: {AVAILABLE_MODELS})")

    if name in ("mel_band_roformer", "bs_roformer"):
        model = _load_msst_model(
            model_type=name,
            msst_dir=model_cfg.msst_dir,
            config_path=model_cfg.msst_config,
            checkpoint=model_cfg.checkpoint,
        )
    else:
        # htdemucs — 레거시 비교용. config의 checkpoint는 RoFormer용이므로
        # 무시하고 항상 demucs 기본 사전학습 가중치를 사용한다.
        model = _load_htdemucs()

    if device is not None:
        model = model.to(device)
    return model


def _load_msst_model(
    model_type: str,
    msst_dir: str,
    config_path: str,
    checkpoint: str | None,
) -> nn.Module:
    """MSST(ZFTurbo/Music-Source-Separation-Training)의 구현으로 모델 생성."""
    msst_path = Path(msst_dir)
    if not msst_path.exists():
        raise FileNotFoundError(
            f"MSST 체크아웃이 없습니다: {msst_path}\n"
            "먼저 `bash scripts/setup_msst.sh` 를 실행하세요."
        )
    if str(msst_path.resolve()) not in sys.path:
        sys.path.insert(0, str(msst_path.resolve()))

    get_model_from_config = _import_msst_model_factory()
    model, _config = get_model_from_config(model_type, str(config_path))

    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"체크포인트가 없습니다: {ckpt_path}\n"
                "`bash scripts/setup_msst.sh` 로 Kim 체크포인트를 내려받으세요."
            )
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        # MSST 체크포인트는 순수 state_dict 또는 {'state_dict'|'state': ...} 래핑
        for key in ("state_dict", "state"):
            if isinstance(state, dict) and key in state:
                state = state[key]
                break
        model.load_state_dict(state)

    model.eval()
    return model


def _import_msst_model_factory():
    """MSST 버전에 따라 get_model_from_config 위치가 다르므로 순차 시도."""
    errors = []
    for module_name in ("utils.settings", "utils.model_utils", "utils"):
        try:
            module = __import__(module_name, fromlist=["get_model_from_config"])
            return module.get_model_from_config
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}: {exc}")
    raise ImportError(
        "MSST에서 get_model_from_config를 찾지 못했습니다. "
        "MSST 리포 버전을 확인하세요.\n" + "\n".join(errors)
    )


def _load_htdemucs() -> nn.Module:
    """HTDemucs 로드 — 레거시 베이스라인 비교 전용 (사전학습 가중치 고정)."""
    try:
        from demucs.pretrained import get_model
    except ImportError as exc:
        raise ImportError(
            "demucs가 설치되지 않았습니다. 비교 실험 시에만 `pip install demucs`."
        ) from exc

    model = get_model("htdemucs")
    model.eval()
    return model
