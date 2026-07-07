# Music Source Separation Pipeline (음원 분리 파이프라인)

2026 세종대학교 하계 창의학기제(집중이수제) 프로젝트.

**Mel-Band RoFormer** (Kim 체크포인트, MIT) 기반 보컬 제거 파이프라인.
한정된 로컬 GPU(VRAM) 환경에서 청크 분할 추론 + Overlap-Add로 안정적으로 구동하고,
운영자가 수집한 **(원본 Mix, 반주 Inst) 쌍만으로** 파인튜닝까지 수행하는 것이 목표.

> **모델 선정 근거와 파이프라인 상세 설계: [docs/PIPELINE_DESIGN.md](docs/PIPELINE_DESIGN.md)**
> (2026-07 MVSep 리더보드·MSST 프레임워크·라이선스 리서치 기반 재설계)

## 아키텍처 요약

- **모델**: Mel-Band RoFormer — 2026-07 현재 보컬 분리 리더보드는 RoFormer 계열이 독점.
  Kim 체크포인트(multisong SDR ~11.0)는 상위권 공개 모델 중 유일한 MIT 라이선스.
- **프레임워크**: [ZFTurbo MSST](https://github.com/ZFTurbo/Music-Source-Separation-Training) —
  2-stem 학습, LoRA, VRAM 절감 옵션 내장. `scripts/setup_msst.sh`로 설치.
- **2-stem 학습 매핑**: `other.wav = 반주`, `vocals.wav = mix − inst` (샘플 정렬 감산).
  단일 타깃(vocals) 모델이므로 추론 시 반주는 `mixture − vocals`로 정확히 복원.
- **추론**: 8초 청크, 75% 오버랩(step=chunk/4), 크로스페이드 Overlap-Add → OOM 없이 처리.

## 환경 설정

```bash
conda activate changuihakgi
pip install -r requirements.txt
bash scripts/setup_msst.sh    # MSST 클론 + Kim 체크포인트(~913MB) 다운로드
```

## 디렉토리 구조

```
creative_semester/
├── docs/PIPELINE_DESIGN.md   # ★ 재설계 문서 (모델 선정 근거·파라미터·함정 목록)
├── config/default.yaml       # 파이프라인 설정
├── data/
│   ├── raw/                  # 수집 데이터: <곡명>/{mix.*, inst.*}
│   ├── processed/            # 학습셋: <곡명>/{vocals.wav, other.wav} (MSST 레이아웃)
│   └── valid/                # 홀드아웃 검증셋 (≥5곡)
├── models/checkpoints/       # Kim 시작 체크포인트 + 파인튜닝 결과
├── third_party/              # MSST 체크아웃 (gitignore, 스크립트로 재현)
├── src/
│   ├── audio/                # I/O(io.py), 청크+Overlap-Add(chunking.py)
│   ├── models/registry.py    # Mel-Band RoFormer 로딩 (MSST 경유)
│   ├── inference/pipeline.py # Mode 1: 추론 파이프라인
│   ├── training/             # Mode 2: 전처리·Dataset·손실·자체 학습 루프
│   └── utils/
├── app/webui.py              # Gradio Web UI
├── scripts/                  # setup_msst / separate / prepare_data / train
└── tests/
```

## 사용법

```bash
# 추론 (보컬/반주 분리)
python scripts/separate.py --input song.mp3 --output outputs/

# 데이터 전처리: (mix, inst) 쌍 -> MSST 학습 레이아웃
python scripts/prepare_data.py --config config/default.yaml

# 파인튜닝 A: MSST train.py (검증된 기본 경로 — 명령어는 설계 문서 Mode 2 참고)
# 파인튜닝 B: 자체 학습 루프 (창의학기제 학습 목표)
python scripts/train.py --config config/default.yaml

# Web UI
python app/webui.py

# 테스트
python -m pytest tests/ -v
```

## 역할 분담

- **이준영** — 모델 분석·파이프라인 구축, 청크 분할 추론 + Overlap-Add, 자체 학습 루프
- **안준석** — 오디오 I/O·전처리(공통 게인·감산 유도), Dataset/DataLoader, Web UI 연동

## 진행 일정

| 주차 | 일시 | 내용 |
|------|------|------|
| 1 | 6/26 | SOTA 논문 리뷰 / 오픈소스 아키텍처·메모리 한계 분석 |
| 2 | 6/29 | 로컬 환경 세팅 / 베이스라인 구동 / OOM 한계 테스트 |
| 3 | 7/6  | 분할 추론 아키텍처 설계 / 크로스페이드(Overlap-Add) 구현 ✅ + SOTA 재조사·파이프라인 재설계 ✅ |
| 4 | 7/13 | 오디오 I/O 모듈 / Kim 체크포인트 1차 추론 파이프라인 완성 |
| 5 | 7/20 | 커스텀 데이터 전처리 / Dataset·DataLoader |
| 6 | 7/27 | Fine-tuning (MSST + 자체 루프) / 홀드아웃 SDR 검증 |
| 7 | 8/3  | Web UI 개발 / 백엔드 연동 |
| 8 | 8/7  | 통합 디버깅 / 문서화·최종 보고서 |
