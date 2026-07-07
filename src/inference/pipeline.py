"""추론 파이프라인 (Mode 1).

[입력/전처리 -> 청크 분할 추론 -> Overlap-Add 병합 -> 감산 복원 -> 출력]

주력 모델(Mel-Band RoFormer, Kim)은 단일 타깃(vocals) 모델이므로
반주는 instrumental = mixture − vocals 감산으로 정확히 복원한다.
(MSST inference.py의 --extract_instrumental 패턴과 동일)
"""
from __future__ import annotations

from pathlib import Path

import torch

from src.audio.chunking import chunked_inference
from src.audio.io import load_audio, save_audio
from src.models.registry import load_model
from src.utils.config import Config
from src.utils.device import get_device


class SeparationPipeline:
    """음원 분리 추론 파이프라인."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = get_device(cfg.inference.device)
        self.model = load_model(cfg.model, device=self.device)
        self.sample_rate = cfg.audio.sample_rate

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

    def _model_forward(self, chunk: torch.Tensor) -> torch.Tensor:
        """모델에 청크를 통과시킨다 (배치 차원 추가/제거)."""
        with torch.no_grad():
            out = self.model(chunk.unsqueeze(0))
        # 단일 타깃: [1, C, T] / 멀티 스템: [1, S, C, T]
        return out.squeeze(0)

    def _to_stem_dict(
        self, estimates: torch.Tensor, mixture: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """모델 출력 -> {stem 이름: waveform}. 단일 타깃이면 감산으로 반주 복원."""
        if estimates.dim() == 2:  # [C, T] — 단일 타깃(vocals) 모델
            vocals = estimates
            return {"vocals": vocals, "instrumental": mixture - vocals}

        # [S, C, T] — 멀티 스템 모델 (htdemucs 레거시 등)
        sources = getattr(self.model, "sources", None)
        if sources:
            stems = {name: estimates[i] for i, name in enumerate(sources)}
            if "vocals" in stems and "instrumental" not in stems:
                stems["instrumental"] = mixture - stems["vocals"]
            return stems
        return {f"stem_{i}": estimates[i] for i in range(estimates.shape[0])}
