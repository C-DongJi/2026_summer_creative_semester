"""Web UI (담당: 안준석).

Gradio 기반 인터페이스: 오디오 업로드 -> 분리 -> Vocal/Inst 다운로드.
(설계서 사용자 환경 - Web UI)

실행:
    python app/webui.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402

from src.inference.pipeline import SeparationPipeline  # noqa: E402
from src.utils.config import load_config  # noqa: E402

CONFIG_PATH = "config/default.yaml"
OUTPUT_DIR = "outputs"

# 파이프라인은 무거우므로 1회 로드 후 재사용 (지연 초기화)
_pipeline: SeparationPipeline | None = None


def get_pipeline() -> SeparationPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SeparationPipeline(load_config(CONFIG_PATH))
    return _pipeline


def separate(audio_path: str):
    """업로드된 오디오를 분리해 stem 파일 경로들을 반환한다."""
    if not audio_path:
        return None, None
    pipeline = get_pipeline()
    paths = pipeline.separate_to_files(audio_path, OUTPUT_DIR)
    vocals = paths.get("vocals")
    inst = paths.get("instrumental") or paths.get("inst")
    return (str(vocals) if vocals else None,
            str(inst) if inst else None)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="음원 분리 파이프라인") as demo:
        gr.Markdown("# 🎵 음원 분리 (Music Source Separation)\nMP3/WAV 업로드 → 보컬/반주 분리")
        with gr.Row():
            inp = gr.Audio(type="filepath", label="음원 업로드 (MP3/WAV)")
        btn = gr.Button("분리 시작", variant="primary")
        with gr.Row():
            out_vocals = gr.Audio(label="Vocal (보컬)")
            out_inst = gr.Audio(label="Inst (반주)")
        btn.click(separate, inputs=inp, outputs=[out_vocals, out_inst])
    return demo


if __name__ == "__main__":
    build_ui().launch()
