"""오디오 I/O 검증 스크립트 (4주차 로컬 검증, GPU 불필요).

다양한 포맷/샘플레이트/채널의 파일이 모델 입력 규격(44.1kHz 스테레오)으로
정확히 변환되는지 검증한다. 실제 파일을 주면 그 파일로, 없으면 합성
사인파로 여러 케이스를 만들어 자동 검증한다.

사용 예:
    python scripts/verify_audio_io.py                     # 합성 케이스 자동 검증
    python scripts/verify_audio_io.py --input song.mp3    # 실제 파일 검증
"""
from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from src.audio.io import load_audio, save_audio  # noqa: E402

TARGET_SR = 44100
TARGET_CH = 2


def make_sine(sr: int, channels: int, seconds: float, freq: float = 440.0) -> torch.Tensor:
    """검증용 사인파 [channels, samples] 생성."""
    t = torch.arange(int(sr * seconds)) / sr
    wave = 0.5 * torch.sin(2 * math.pi * freq * t)
    return wave.unsqueeze(0).repeat(channels, 1)


def verify_file(path: Path, expect_seconds: float | None = None) -> dict:
    """파일 하나를 로드해 규격 변환을 검증."""
    r: dict = {"file": path.name, "issues": []}
    try:
        wav, sr = load_audio(path, TARGET_SR, TARGET_CH)
    except Exception as exc:
        r["issues"].append(f"load_error: {type(exc).__name__}: {exc}")
        return r

    r["out_sr"] = sr
    r["out_shape"] = tuple(wav.shape)
    r["duration"] = wav.shape[-1] / sr

    if sr != TARGET_SR:
        r["issues"].append(f"sr={sr} (기대 {TARGET_SR})")
    if wav.shape[0] != TARGET_CH:
        r["issues"].append(f"channels={wav.shape[0]} (기대 {TARGET_CH})")
    if not torch.isfinite(wav).all():
        r["issues"].append("NaN/Inf 존재")
    if wav.abs().max() < 1e-6:
        r["issues"].append("무음 (디코딩 실패 의심)")
    if expect_seconds is not None and abs(r["duration"] - expect_seconds) > 0.05:
        r["issues"].append(f"길이 {r['duration']:.3f}s (기대 {expect_seconds:.3f}s)")
    return r


def synthetic_cases(tmp: Path) -> list[tuple[Path, float]]:
    """포맷·샘플레이트·채널 조합의 합성 테스트 파일들 생성."""
    seconds = 2.0
    cases: list[tuple[Path, float]] = []

    specs = [
        ("wav_44k_stereo.wav", 44100, 2, "PCM_16"),
        ("wav_48k_stereo.wav", 48000, 2, "PCM_16"),
        ("wav_22k_mono.wav", 22050, 1, "PCM_16"),
        ("wav_44k_float32.wav", 44100, 2, "FLOAT"),
        ("flac_48k_stereo.flac", 48000, 2, None),
        ("mp3_44k_stereo.mp3", 44100, 2, None),  # libsndfile 1.1+ 필요
        ("ogg_44k_stereo.ogg", 44100, 2, None),
    ]
    for name, sr, ch, subtype in specs:
        path = tmp / name
        wave = make_sine(sr, ch, seconds)
        try:
            sf.write(str(path), wave.numpy().T, sr, subtype=subtype)
            cases.append((path, seconds))
        except Exception as exc:
            print(f"  (스킵) {name}: 이 환경에서 생성 불가 — {type(exc).__name__}")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="오디오 I/O 규격 변환 검증")
    parser.add_argument("--input", nargs="*", default=None, help="검증할 실제 오디오 파일들")
    args = parser.parse_args()

    print(f"목표 규격: {TARGET_SR} Hz / {TARGET_CH}ch\n")
    results: list[dict] = []

    if args.input:
        for f in args.input:
            results.append(verify_file(Path(f)))
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            print("합성 테스트 파일 생성 중...")
            for path, expect in synthetic_cases(tmp):
                results.append(verify_file(path, expect))

            # 저장 roundtrip: 44.1k 결과를 저장 후 재로드
            wav, _ = load_audio(tmp / "wav_48k_stereo.wav", TARGET_SR, TARGET_CH)
            rt = tmp / "roundtrip.wav"
            save_audio(rt, wav, TARGET_SR, float32=True)
            wav2, _ = load_audio(rt, TARGET_SR, TARGET_CH)
            ok = torch.allclose(wav, wav2, atol=1e-6)
            results.append({
                "file": "roundtrip(float32 save->load)",
                "out_sr": TARGET_SR, "out_shape": tuple(wav2.shape),
                "duration": wav2.shape[-1] / TARGET_SR,
                "issues": [] if ok else ["roundtrip 불일치"],
            })

    print(f"\n{'파일':<32} {'출력':<20} {'길이':>7}  결과")
    print("-" * 75)
    n_bad = 0
    for r in results:
        shape = str(r.get("out_shape", "-"))
        dur = f"{r.get('duration', 0):.2f}s" if "duration" in r else "-"
        status = "OK" if not r["issues"] else "FAIL: " + "; ".join(r["issues"])
        n_bad += bool(r["issues"])
        print(f"{r['file']:<32} {shape:<20} {dur:>7}  {status}")

    print("-" * 75)
    print(f"통과 {len(results) - n_bad} / {len(results)}")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
