"""Web UI (담당: 안준석).

[창의학기제 7주차 (8/3) 산출물 — Web UI 개발 및 백엔드 연동]

Gradio 기반 인터페이스:
  탭 1 음원 분리 — 오디오 업로드 -> 분리 -> Vocal/Inst 다운로드,
                   체크포인트(기본/파인튜닝) 선택 가능
  탭 2 재학습   — 사용자별로 (mix, inst) 쌍 파일을 업로드해 데이터셋을 쌓고,
                   웹에서 바로 파인튜닝 실행 -> 결과 체크포인트를 탭 1에서 사용

실행:
    python app/webui.py                       # 로컬 (127.0.0.1:7860)
    python app/webui.py --host 0.0.0.0        # 같은 네트워크의 다른 기기에서 접속
    python app/webui.py --share               # gradio 임시 공개 링크
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
import torch  # noqa: E402

from src.audio.io import load_audio  # noqa: E402
from src.data.ingest_common import save_stem_pair  # noqa: E402
from src.inference.pipeline import SeparationPipeline  # noqa: E402
from src.training.preprocess import derive_stems  # noqa: E402
from src.training.train import train as run_train  # noqa: E402
from src.utils.config import load_config  # noqa: E402

CONFIG_PATH = ROOT / "config" / "default.yaml"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR = ROOT / "models" / "checkpoints"
USERS_DIR = ROOT / "data" / "users"

# 파이프라인은 무거우므로(가중치 ~0.9GB) 마지막 1개만 캐시
_pipeline_cache: dict[str, SeparationPipeline] = {}


# ---------- 순수 헬퍼 (테스트 대상) ----------

def sanitize_user(name: str | None) -> str:
    """사용자 이름을 디렉토리에 안전한 형태로 정리."""
    return re.sub(r"[^0-9A-Za-z가-힣_-]", "", (name or "").strip())[:40]


def pair_uploads(paths: list[str]) -> tuple[dict[str, tuple[Path, Path]], list[str]]:
    """업로드 파일명 규칙 <곡명>_mix.* / <곡명>_inst.* 로 쌍을 맺는다.

    Returns:
        ({곡명: (mix경로, inst경로)}, 짝이 없는 곡명 목록)
    """
    mixes: dict[str, Path] = {}
    insts: dict[str, Path] = {}
    for p in map(Path, paths or []):
        stem = p.stem
        low = stem.lower()
        if low.endswith("_mix"):
            mixes[stem[:-4]] = p
        elif low.endswith("_inst"):
            insts[stem[:-5]] = p
    pairs = {k: (mixes[k], insts[k]) for k in mixes if k in insts}
    incomplete = sorted((set(mixes) | set(insts)) - set(pairs))
    return pairs, incomplete


# ---------- 체크포인트 / 파이프라인 ----------

def list_checkpoints() -> list[tuple[str, str]]:
    """(표시명, 경로) 목록 — 기본 체크포인트 + 사용자별 파인튜닝 결과."""
    items: list[tuple[str, str]] = []
    base = CKPT_DIR / "MelBandRoformer.ckpt"
    if base.exists():
        items.append(("기본 모델 (Kim 사전학습)", str(base)))
    for p in sorted(CKPT_DIR.glob("finetune_epoch*.ckpt")):
        items.append((f"파인튜닝: {p.name}", str(p)))
    # MSST train.py 결과물: <results_path>/model_<type>_ep_<N>_..._.ckpt
    for p in sorted(CKPT_DIR.glob("*/model_*.ckpt")):
        if p.parent.name == "users":
            continue
        items.append((f"파인튜닝: {p.parent.name}/{p.name}", str(p)))
    for p in sorted(CKPT_DIR.glob("users/*/finetune_epoch*.ckpt")):
        items.append((f"{p.parent.name}님의 모델: {p.name}", str(p)))
    return items


def get_pipeline(ckpt_path: str) -> SeparationPipeline:
    if ckpt_path not in _pipeline_cache:
        _pipeline_cache.clear()  # 이전 모델 메모리 해제 (1개만 유지)
        cfg = load_config(CONFIG_PATH)
        cfg["model"]["checkpoint"] = ckpt_path
        _pipeline_cache[ckpt_path] = SeparationPipeline(cfg)
    return _pipeline_cache[ckpt_path]


# ---------- 탭 1: 분리 ----------

def separate(audio_path: str, ckpt_path: str):
    if not audio_path:
        return None, None, "오디오 파일을 먼저 업로드하세요."
    if not ckpt_path:
        return None, None, "체크포인트를 선택하세요."
    pipeline = get_pipeline(ckpt_path)

    # 실행마다 고유 폴더에 저장 — 이전 실행 결과 덮어쓰기/혼동 방지
    run_dir = OUTPUT_DIR / f"run_{uuid.uuid4().hex[:8]}"
    # 화면의 경과 시간은 '대기열에서 기다린 시간'까지 합산되어 보일 수 있으므로
    # 순수 분리 시간은 여기서 직접 측정해 표시한다.
    t0 = time.time()
    paths = pipeline.separate_to_files(audio_path, run_dir)
    elapsed = time.time() - t0

    note = "" if torch.cuda.is_available() else " · CPU 모드 — 긴 곡은 오래 걸립니다"
    return (
        str(paths["vocals"]),
        str(paths["instrumental"]),
        f"분리 완료 — 이번 작업 **{elapsed:.1f}초** (앞 작업 대기 시간 제외{note})",
    )


# ---------- 탭 2: 재학습 ----------

def add_training_data(user_name: str, files: list[str]):
    user = sanitize_user(user_name)
    if not user:
        return "사용자 이름을 입력하세요."
    if not files:
        return "파일을 업로드하세요. 파일명 규칙: <곡명>_mix.wav / <곡명>_inst.wav"

    cfg = load_config(CONFIG_PATH)
    sr, ch = cfg.audio.sample_rate, cfg.audio.channels
    dest = USERS_DIR / user / "processed"

    pairs, incomplete = pair_uploads(files)
    lines: list[str] = []
    added = 0
    for song, (mix_path, inst_path) in pairs.items():
        mix, _ = load_audio(mix_path, sr, ch)
        inst, _ = load_audio(inst_path, sr, ch)
        stems = derive_stems(mix, inst, sr,
                             normalize=cfg.data.normalize,
                             target_peak=cfg.data.target_peak)
        if stems is None:
            lines.append(f"- ❌ {song}: mix/inst 길이 불일치(0.5초 초과) — 건너뜀")
            continue
        vocals, other = stems
        save_stem_pair(dest / song, vocals, other, sr)
        lines.append(f"- ✅ {song}")
        added += 1
    for name in incomplete:
        lines.append(f"- ⚠️ {name}: 짝(_mix/_inst)이 없어 건너뜀")

    total = len([p for p in dest.iterdir() if p.is_dir()]) if dest.exists() else 0
    lines.append(f"\n**{added}곡 추가 — {user}님의 데이터셋: 총 {total}곡** "
                 "(커뮤니티 기준 ~40곡부터 유의미)")
    return "\n".join(lines)


def start_finetune(user_name: str, epochs: float, lr: float, base_ckpt: str,
                   progress=gr.Progress()):
    user = sanitize_user(user_name)
    if not user:
        return "사용자 이름을 입력하세요.", gr.update(), gr.update()
    user_proc = USERS_DIR / user / "processed"
    n_tracks = len([p for p in user_proc.iterdir() if p.is_dir()]) if user_proc.exists() else 0
    if n_tracks == 0:
        return f"{user}님의 학습 데이터가 없습니다. 먼저 파일을 추가하세요.", gr.update(), gr.update()
    if not base_ckpt:
        return "시작 체크포인트를 선택하세요.", gr.update(), gr.update()

    cfg = load_config(CONFIG_PATH)
    cfg["data"]["processed_dir"] = str(user_proc)
    cfg["training"]["epochs"] = int(epochs)
    cfg["training"]["learning_rate"] = float(lr)
    cfg["training"]["checkpoint_dir"] = str(CKPT_DIR / "users" / user)
    cfg["training"]["resume_from"] = base_ckpt
    cfg["training"]["num_workers"] = 2

    device_note = "GPU" if torch.cuda.is_available() else "CPU(느림 — GPU PC 권장)"
    progress(0.0, desc=f"{device_note}에서 학습 시작 ({n_tracks}곡, {int(epochs)}에폭)")

    def on_epoch_end(epoch: int, avg_loss: float) -> None:
        progress(epoch / int(epochs), desc=f"epoch {epoch}/{int(epochs)} loss {avg_loss:.4f}")

    last_ckpt = run_train(cfg, on_epoch_end=on_epoch_end)

    choices = list_checkpoints()
    msg = (f"✅ 재학습 완료 ({device_note}, {n_tracks}곡 × {int(epochs)}에폭)\n\n"
           f"저장: `{last_ckpt}`\n\n'음원 분리' 탭에서 방금 만든 모델을 선택해 사용해보세요.")
    return msg, gr.update(choices=choices), gr.update(choices=choices)


# ---------- UI ----------

def build_ui() -> gr.Blocks:
    ckpts = list_checkpoints()
    default_ckpt = ckpts[0][1] if ckpts else None

    with gr.Blocks(title="음원 분리 파이프라인") as demo:
        gr.Markdown("# 음원 분리 (Music Source Separation)\n"
                    "MP3/WAV 업로드 → 보컬/반주 분리. 재학습 탭에서 나만의 모델을 만들 수 있습니다.")

        with gr.Tab("음원 분리"):
            ckpt_dd = gr.Dropdown(choices=ckpts, value=default_ckpt, label="사용할 모델")
            inp = gr.Audio(type="filepath", label="음원 업로드 (MP3/WAV)")
            sep_btn = gr.Button("분리 시작", variant="primary")
            sep_status = gr.Markdown()
            with gr.Row():
                out_vocals = gr.Audio(label="Vocal (보컬)")
                out_inst = gr.Audio(label="Inst (반주)")
            sep_btn.click(separate, inputs=[inp, ckpt_dd],
                          outputs=[out_vocals, out_inst, sep_status])

        with gr.Tab("재학습 (Fine-tuning)"):
            gr.Markdown(
                "### 나만의 데이터로 모델 만들기\n"
                "1. 파일명을 `<곡명>_mix.wav`(원본) / `<곡명>_inst.wav`(반주) 규칙으로 준비\n"
                "2. 사용자 이름 입력 후 파일 추가 — 여러 번 나눠 올려도 누적됩니다\n"
                "3. 재학습 시작 → 완료되면 '음원 분리' 탭에서 내 모델 선택\n"
            )
            user_tb = gr.Textbox(label="사용자 이름", placeholder="예: leejy")
            up_files = gr.File(file_count="multiple", type="filepath",
                               file_types=["audio"], label="(mix, inst) 쌍 파일들")
            add_btn = gr.Button("데이터 추가")
            add_status = gr.Markdown()
            add_btn.click(add_training_data, inputs=[user_tb, up_files],
                          outputs=[add_status])

            with gr.Row():
                epochs_sl = gr.Slider(1, 50, value=5, step=1, label="에폭 수")
                lr_num = gr.Number(value=1e-5, label="learning rate")
            base_dd = gr.Dropdown(choices=ckpts, value=default_ckpt,
                                  label="시작 체크포인트 (보통 기본 모델)")
            train_btn = gr.Button("재학습 시작", variant="primary")
            train_status = gr.Markdown()
            train_btn.click(
                start_finetune,
                inputs=[user_tb, epochs_sl, lr_num, base_dd],
                outputs=[train_status, ckpt_dd, base_dd],
            )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="음원 분리 Web UI")
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0이면 같은 네트워크의 다른 기기에서 접속 가능")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="gradio 임시 공개 링크 생성")
    args = parser.parse_args()
    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share)
