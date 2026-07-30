"""추론 파이프라인 (Mode 1).

[창의학기제 4주차 (7/13) 산출물 — 1차 추론 파이프라인 완성]

[입력/전처리 -> 청크 분할 추론 -> Overlap-Add 병합 -> 감산 복원 -> 출력]

주력 모델(Mel-Band RoFormer, Kim)은 단일 타깃(vocals) 모델이므로
반주는 instrumental = mixture − vocals 감산으로 정확히 복원한다.
(MSST inference.py의 --extract_instrumental 패턴과 동일)

경량화: CUDA에서는 config inference.precision(auto|fp16|bf16|fp32)에 따라
autocast 반정밀 추론을 사용한다 (VRAM 약 절반, 1.5~2배 속도).
Overlap-Add 누적은 항상 fp32로 수행해 정밀도를 유지한다.
"""
from __future__ import annotations

from pathlib import Path

import torch

from src.audio.chunking import chunked_inference
from src.audio.io import load_audio, save_audio
from src.models.registry import load_model
from src.utils.config import Config
from src.utils.device import get_device


def _resolve_autocast_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    """precision 설정 -> autocast dtype (None이면 fp32 유지)."""
    if device.type != "cuda":
        return None  # CPU autocast는 이득이 없어 fp32 유지
    precision = (precision or "auto").lower()
    if precision == "fp32":
        return None
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    # auto: bf16 지원 GPU(Ampere+)면 bf16(수치 안정), 아니면 fp16(T4 등)
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class SeparationPipeline:
    """음원 분리 추론 파이프라인."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = get_device(cfg.inference.device)
        self.model = load_model(cfg.model, device=self.device)
        self.sample_rate = cfg.audio.sample_rate
        self.autocast_dtype = _resolve_autocast_dtype(
            getattr(cfg.inference, "precision", "auto"), self.device
        )
        self.batch_size = max(1, int(getattr(cfg.inference, "batch_size", 1)))

    def separate(self, input_path: str | Path) -> dict[str, torch.Tensor]:
        """오디오 파일을 vocals / instrumental로 분리한다.

        Returns:
            {"vocals": [C, T], "instrumental": [C, T]}
        """
        mixture, sr = load_audio(
            input_path,
            target_sr=self.sample_rate,
            target_channels=self.cfg.audio.channels,
        )

        chunk_samples = int(self.cfg.inference.chunk_seconds * sr)
        estimates = chunked_inference(
            mixture,
            process_fn=self._model_forward,
            chunk_samples=chunk_samples,
            overlap=self.cfg.inference.overlap,
            fade=self.cfg.inference.fade,
            device=self.device,
            batch_size=self.batch_size,
        )
        return self._to_stem_dict(estimates, mixture)

    def separate_to_files(
        self, input_path: str | Path, output_dir: str | Path
    ) -> dict[str, Path]:
        """분리 결과를 파일로 저장하고 경로를 반환한다."""
        output_dir = Path(output_dir)
        stems = self.separate(input_path)
        paths: dict[str, Path] = {}
        stem_name = Path(input_path).stem
        for name, wav in stems.items():
            out = output_dir / f"{stem_name}_{name}.wav"
            save_audio(out, wav, self.sample_rate)
            paths[name] = out
        return paths

    def _model_forward(self, batch: torch.Tensor) -> torch.Tensor:
        """배치 청크 [B, C, T] -> 추정 [B, (S,) C, T]. 반정밀 추론 지원."""
        with torch.inference_mode():
            if self.autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    out = self.model(batch)
                return out.float()  # Overlap-Add 누적은 fp32 유지
            return self.model(batch)

    def _to_stem_dict(
        self, estimates: torch.Tensor, mixture: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """모델 출력 -> {stem 이름: waveform}. 단일 타깃이면 감산으로 반주 복원."""
        if estimates.dim() == 2:  # [C, T] — 단일 타깃(vocals) 모델
            vocals = estimates
            return {"vocals": vocals, "instrumental": mixture - vocals}

        # [S, C, T] — 멀티 스템 모델 (htdemucs 레거시 비교 전용.
        # 주의: demucs의 정식 추론 API는 apply_model이며 raw forward는
        # 세그먼트 정규화가 빠져 품질이 다를 수 있다 — 비교 실험 참고용)
        sources = getattr(self.model, "sources", None)
        if sources:
            stems = {name: estimates[i] for i, name in enumerate(sources)}
            if "vocals" in stems and "instrumental" not in stems:
                stems["instrumental"] = mixture - stems["vocals"]
            return stems
        return {f"stem_{i}": estimates[i] for i in range(estimates.shape[0])}
