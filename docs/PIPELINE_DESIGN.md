# 음원 분리 파이프라인 재설계 (2026-07 기준 SOTA)

> 2026-07-07 웹 리서치(MVSep 리더보드, ZFTurbo MSST 리포, HuggingFace, arXiv)로 검증한 사실 기반.
> 제약 조건: **파인튜닝 데이터는 (원본 Mix, 반주 Inst) 쌍만 사용 가능** — 악기별 멀티 스템 데이터셋 사용 불가.

## 0. 결정 요약

| 항목 | 결정 | 근거 |
|------|------|------|
| 아키텍처 | **Mel-Band RoFormer** (band-split RoPE transformer) | 보컬 타깃에서 BS-RoFormer 대비 +0.43~0.58dB (arXiv 2310.01809) |
| 시작 체크포인트 | **Kim (KimberleyJSN/melbandroformer)** — MelBandRoformer.ckpt | 상위권 보컬 모델 중 **유일한 MIT 라이선스**, multisong SDR ~11.0 |
| 학습/추론 프레임워크 | **ZFTurbo/Music-Source-Separation-Training (MSST)** | MIT, 활발히 유지보수(2026-07-05 push), 2-stem 학습·LoRA·VRAM 절감 옵션 내장 |
| 학습 방식 | 전체 파인튜닝(기본) / **LoRA**(8GB VRAM 폴백) | unwa가 동일 체크포인트를 RTX 3060 Ti 8GB로 파인튜닝한 선례 |
| 추론 방식 | 8초 청크 + 75% 오버랩(step=chunk/4) Overlap-Add | MSST demix()와 동일 규격, 커뮤니티 측정상 overlap 4 이상은 이득 <0.01dB |
| 반주 복원 | **instrumental = mixture − predicted vocals** | 단일 타깃(vocals) 모델이므로 감산으로 정확히 복원 |

## 1. 2026-07 리더보드 현황 (MVSep multisong, 보컬 SDR 기준)

MVSep Quality Checker(100곡×1분, 44.1kHz)는 사실상의 커뮤니티 표준 벤치마크다.

| 순위 | 모델 | Vocals SDR | 공개 여부 | 계열 |
|------|------|-----------|----------|------|
| 1 | BS PolarFormer 124 bands (2026-06) | 12.02 | ❌ MVSep 내부 전용 | BS-RoFormer + PoPE |
| 2 | BS Roformer ver.2025.07 (MVSep) | 11.89 | ❌ 사이트 전용 | BS-RoFormer |
| 3 | sami-bytedance v1.1 | 11.82 | ❌ 상용 API | BS-RoFormer (원저자) |
| 4 | **unwa Leap Xe / Leap** | 11.72~11.76 | ⭕ HF 공개, **라이선스 미표기** | BS-RoFormer 파인튜닝 |
| … | becruily deux | 11.37 | ⭕ 공개, **CC-BY-NC** | Mel-Band RoFormer |
| … | BS PolarFormer 공개판 (60 bands) | 11.00 | ⭕ 공개, fp16 전용 | PolarFormer |
| … | **Kim MelBandRoformer** ★채택 | **10.98~11.03** | ⭕ 공개, **MIT** | Mel-Band RoFormer |
| … | viperx bs_roformer ep_317 | 10.87 | ⭕ 공개, 라이선스 미표기 | BS-RoFormer |
| 참고 | HTDemucs (v4) | ~3-4dB 뒤처짐 | ⭕ | 하이브리드 CNN+Transformer |

핵심 관찰:
- **상위 10개가 전부 RoFormer 계열.** SCNet(9.43), BS-Mamba2(8.83, multisong), HTDemucs는 실전 벤치마크에서 크게 뒤처진다.
- 2026년 신규 아키텍처(Diff-VS, discrete-token LM, Moises-Light, SFC)는 **공개 보컬 가중치가 없어** 파인튜닝 시작점이 될 수 없다.
- 최고 단일 모델(12.02)이 최고 앙상블(11.93)을 이미 추월 — 단일 모델 파이프라인으로 충분한 시대.

## 2. 왜 Kim Mel-Band RoFormer인가 (선정 기준 5단계)

1. **하드 제약 충족**: MSST에서 `instruments: [vocals, other]`, `target_instrument: vocals`(단일 타깃 2-stem)로 학습하는 것이 상위권 보컬 모델 전원의 표준 학습법. 우리 데이터는 `other.wav = 반주`, `vocals.wav = mix − inst`(샘플 정렬 감산)로 정확히 매핑된다. MSST 데이터 로더는 `mixture.wav`가 없으면 스템 합으로 재구성하므로 (mix, inst) 쌍만으로 충분.
2. **품질**: Mel-Band RoFormer는 보컬 타깃에서 BS-RoFormer 대비 +0.43~0.58dB(동일 깊이, arXiv 2310.01809 Table 1). Kim 체크포인트는 공개 모델 중 상위권(~11.0).
3. **라이선스**: 상위권 공개 체크포인트 중 **유일하게 명시적 MIT**. unwa Leap(최고 공개 SDR 11.76)은 라이선스 미표기 = 법적으로 all-rights-reserved. 대학 프로젝트로 파생물(파인튜닝 모델, 리포, 보고서)을 재배포하려면 Kim이 유일한 깨끗한 선택지.
4. **8~16GB VRAM 파인튜닝 검증됨**: 커뮤니티 계보(unwa FT 시리즈, gabox, becruily)가 전부 Kim에서 출발. 특히 unwa는 **RTX 3060 Ti 8GB**에서 chunk 축소 + gradient checkpointing + AdamW8bit으로 41곡 데이터셋 파인튜닝에 성공(~0.1 SDR 손실). 정확한 학습 config가 MSST에 동봉(`configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml`).
5. **추론 효율**: 8초 청크 기준 6~8GB급 GPU에서 구동. MSST `demix()`의 청크+Overlap-Add 로직은 본 프로젝트가 이미 구현한 것과 동일한 구조.

### 폴백/업그레이드 경로
- **unwa BS-Roformer-Leap** (11.72~11.76, 59.6M 파라미터로 Kim의 1/4): 라이선스 문제만 해소되면 최선의 공개 모델. **추론 전용 A/B 비교**로는 지금도 활용 가능.
- **BS PolarFormer 공개판** (51M, fp16): 최신 아키텍처. fp16 가중치의 파인튜닝 안정성 미검증 → 관찰 대상.
- **SCNet Masked XL**: RoFormer 학습이 불가능할 때의 최후 폴백(비-어텐션, 가장 저렴). 시작점이 1.6dB 뒤처짐.
- **Apollo (JusperLee)**: 분리가 아닌 **복원(restoration)** 모델 — 분리 후 반주 아티팩트 복원용 후처리 단계로 향후 추가 가능.

### 왜 HTDemucs를 버리는가 (원 설계서 대비 변경점)
- facebookresearch/demucs 리포는 **아카이브(읽기 전용)** 상태.
- 사전학습 가중치가 4-stem(drums/bass/other/vocals) 고정 → (mix, inst) 쌍만으로 2-stem 파인튜닝하려면 지원되지 않는 state_dict 수술이 필요. **본 프로젝트의 하드 제약과 정면 충돌.**
- 품질도 RoFormer 대비 3~4dB 뒤처짐. `--two-stems` 플래그는 추론 편의 기능일 뿐 학습 지원이 아님.
- 코드에는 레거시 베이스라인 비교용으로만 남긴다.

## 3. 파이프라인 구조

### Mode 1 — 추론 (보컬 제거)

```
MP3/WAV 입력
  → [전처리] 44.1kHz 스테레오 리샘플 (src/audio/io.py)
  → [청크 분할] 8초(352,800샘플) 청크, step = chunk/4 (75% 오버랩)
  → [모델 추론] Mel-Band RoFormer → 보컬 추정 (청크당 VRAM ~수 GB)
  → [Overlap-Add] 페이드 윈도우 가중 합산 + 가중치 정규화 (src/audio/chunking.py)
  → [반주 복원] instrumental = mixture − vocals   ← 단일 타깃 모델의 정확한 감산
  → vocal.wav + inst.wav 출력
```

- 모델 STFT 규격(체크포인트 고정): n_fft 2048, hop 441, 60 mel bands, 44.1kHz 스테레오.
- MSST 규격: `num_overlap 4`, 선형 페이드(fade_size = chunk/10), 첫/끝 청크 경계는 1로 고정. 본 프로젝트 구현은 Hann 윈도우 + 반사 패딩으로 동일 문제를 해결(정합성 단위테스트 통과).
- overlap을 4→32로 올려도 이득이 +0.06dB 미만(커뮤니티 측정) → 기본 4.

### Mode 2 — 파인튜닝

```
운영자 수집: data/raw/<곡명>/{mix.wav, inst.wav}
  → [전처리] scripts/prepare_data.py
      · 샘플 정렬 검증 (길이 불일치 시 경고/스킵)
      · 공통 게인 정규화 (mix 피크 기준 동일 게인을 양쪽에 적용 — 개별 정규화 금지!)
      · vocals = mix − inst (float32 감산, 32-bit float WAV로 저장)
      → data/processed/<곡명>/{vocals.wav, other.wav}   ← MSST dataset_type 4 레이아웃
  → [학습] 두 가지 경로
      A. MSST train.py (검증된 기본 경로):
         python third_party/Music-Source-Separation-Training/train.py \
           --model_type mel_band_roformer \
           --config_path config/msst_finetune.yaml \
           --start_check_point models/checkpoints/MelBandRoformer.ckpt \
           --dataset_type 4 --data_path data/processed --valid_path data/valid \
           --metrics sdr log_wmse --metric_for_scheduler sdr
      B. 자체 학습 루프 (src/training/train.py — 창의학기제 학습 목표: 루프 직접 구현)
  → [검증] 홀드아웃 ≥5곡(장르 다양성 확보)의 SDR 매 epoch 추적
  → 가중치 업데이트 → Mode 1에 재투입
```

### 하이퍼파라미터 (커뮤니티 수렴값)

| 항목 | 8GB VRAM | 16GB VRAM |
|------|----------|-----------|
| chunk_size | 131,584 (~3.0초) | 352,800 (8초, 네이티브) |
| batch_size | 1 | 1~2 |
| gradient_accumulation | 8 | 4~8 |
| lr | 1e-5 (불안정 시 5e-6) | 1e-5 |
| use_amp / use_torch_checkpoint | 둘 다 켬 | 둘 다 켬 |
| optimizer | **adamw8bit** | adam/adamw |
| LoRA | 필요 시 r=8, alpha=16 (`--train_lora_peft`) | 선택 |

- gradient checkpointing 효과 실측 예: BS-RoFormer 8초/dim256/depth8에서 21GB → 13GB.
- 손실 함수: 체크포인트가 학습된 **multi-resolution STFT loss** 유지 (windows [4096,2048,1024,512,256], hop 147). 초기에 손실 함수를 바꾸지 말 것.
- 스케줄러(ReduceLROnPlateau) patience를 초기에 크게(~1000) — 소규모 데이터에서 조기 lr 붕괴 방지.
- 데이터 규모 기준: 커뮤니티 선례상 **~40곡부터 유의미**, 170~750곡이면 견고.

## 4. 함정 목록 (리서치에서 확인된 실패 사례)

1. **개별 정규화 금지**: mix와 inst를 각각 피크 정규화하면 `vocals = mix − inst` 관계가 깨져 보컬 스템에 반주가 새어 들어간다. 반드시 **동일 게인**을 쌍에 적용.
2. **샘플 정렬**: mix/inst 길이가 다르면 MSST에서 브로드캐스트 에러. 감산 전 정렬·트림 필수. 인코딩 지연(MP3 gapless 등)으로 인한 오프셋 주의 — 감산은 **디코딩 후 float32에서** 수행.
3. **metadata 캐시**: MSST는 `metadata_*.pkl`로 데이터셋을 캐싱 — 데이터 변경 후 삭제하지 않으면 변경이 조용히 무시됨.
4. **검증셋 필수**: unwa의 "무증상 품질 저하" 사례 — 학습 손실은 잘 내려가는데 실제 분리 품질이 망가진 경우가 보고됨. 홀드아웃 SDR 없이는 감지 불가.
5. **python-audio-separator의 overlap 함정**: RoFormer 브랜치에서 overlap 파라미터를 '초 단위 step'으로 해석하며 기본값이 사실상 오버랩 0 — 커스텀 체크포인트 로딩도 불가. **UI의 기성 모델 비교용으로만** 사용.
6. **fp16 체크포인트 파인튜닝**: PolarFormer 공개판처럼 fp16 전용 가중치는 파인튜닝 안정성이 미검증.

## 5. 원 설계서와의 차이 요약

| 원 설계서 (2026-05) | 재설계 (2026-07) | 이유 |
|---|---|---|
| HTDemucs + RoFormer 병용 | **Mel-Band RoFormer 단일** (HTDemucs는 레거시 비교용) | 아카이브된 리포, 4-stem 제약, 3-4dB 열세 |
| 자체 학습 루프만 | **MSST train.py(기본) + 자체 루프(학습 목표용)** 이원화 | 검증된 경로 확보 + 창의학기제 학습 목표 유지 |
| 모델 미지정 파인튜닝 | **Kim ckpt에서 시작** (MIT), LoRA 폴백 | 라이선스·8GB 선례·config 동봉 |
| 청크/오버랩 자체 설계 | MSST demix() 규격에 정합 (8초/overlap 4) | 커뮤니티 실측 기반 파라미터 |
| — | 반주 = mix − vocals 감산 복원 | 단일 타깃 모델 채택에 따른 구조 확정 |

## 6. 참고 링크

- MSST 프레임워크: https://github.com/ZFTurbo/Music-Source-Separation-Training (MIT)
- Kim 체크포인트: https://huggingface.co/KimberleyJSN/melbandroformer (MIT)
- Kim 학습 config: MSST `configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml`
- LoRA 문서: MSST `docs/LoRA.md`
- Mel-Band RoFormer 논문: https://arxiv.org/abs/2310.01809
- BS-RoFormer 논문: https://arxiv.org/abs/2309.02612
- MVSep 리더보드: https://mvsep.com/quality_checker/multisong_leaderboard?sort=vocals
- unwa Leap (A/B 비교용): https://huggingface.co/pcunwa/BS-Roformer-Leap
