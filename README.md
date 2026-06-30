# Music Source Separation Pipeline (음원 분리 파이프라인)

2026 세종대학교 하계 창의학기제(집중이수제) 프로젝트.

HTDemucs, BS-RoFormer, Mel-RoFormer 등 최신 SOTA 음원 분리 모델을 한정된 로컬 GPU(VRAM)
환경에서 안정적으로 구동하기 위한 **실용적 보컬 제거 파이프라인**을 구축한다. 단순 추론을 넘어
커스텀 데이터를 활용한 **재학습(Fine-tuning) 환경**과 **Web UI**까지 통합하는 것을 목표로 한다.

## 핵심 목표

1. **분할 추론(Chunked Inference)** — 긴 음원(3~4분)을 청크로 분할 추론 후 Overlap-Add로 병합하여 OOM 회피
2. **오디오 I/O** — MP3/WAV 등 다양한 포맷·샘플레이트를 모델 입력 규격으로 변환하는 전처리
3. **재학습 파이프라인** — Mix/Inst 음원 쌍을 정규화·노이즈 제거하여 PyTorch Dataset으로 변환, Fine-tuning
4. **Web UI** — Gradio/Streamlit 기반 업로드/다운로드 인터페이스

## 환경 설정

```bash
conda activate changuihakgi
pip install -r requirements.txt
```

## 디렉토리 구조

```
creative_semester/
├── config/             # 설정 파일 (yaml)
├── data/
│   ├── raw/            # 수집한 원본 Mix/Inst 음원
│   └── processed/      # 정규화된 학습용 데이터셋
├── models/checkpoints/ # 사전학습 / 재학습 가중치
├── outputs/            # 분리 결과물 (vocal / inst)
├── notebooks/          # 실험 노트북
├── src/
│   ├── audio/          # 오디오 I/O, 청크 분할 + Overlap-Add
│   ├── models/         # 모델 로딩 (HTDemucs, RoFormer)
│   ├── inference/      # 추론 파이프라인
│   ├── training/       # Dataset/DataLoader, 전처리, Fine-tuning 루프
│   └── utils/          # 공통 유틸 (config, logging)
├── app/                # Web UI (Gradio/Streamlit)
├── scripts/            # CLI 진입점 (separate, prepare_data, train)
└── tests/
```

## 사용법 (개발 중)

```bash
# 추론
python scripts/separate.py --input song.mp3 --output outputs/ --model htdemucs

# 데이터 전처리
python scripts/prepare_data.py --raw data/raw --out data/processed

# 재학습
python scripts/train.py --config config/default.yaml

# Web UI
python app/webui.py
```

## 역할 분담

- **이준영** — SOTA 모델 분석, 파이프라인 구축, 청크 분할 추론 + Overlap-Add, Fine-tuning 학습 루프
- **안준석** — 오디오 I/O·전처리, PyTorch Dataset/DataLoader, Web UI 연동 및 백엔드 통합

## 진행 일정

| 주차 | 일시 | 내용 |
|------|------|------|
| 1 | 6/26 | SOTA 논문 리뷰 / 오픈소스 아키텍처·메모리 한계 분석 |
| 2 | 6/29 | 로컬 환경 세팅 / 베이스라인 구동 / OOM 한계 테스트 |
| 3 | 7/6  | 분할 추론 아키텍처 설계 / 크로스페이드(Overlap-Add) 구현 |
| 4 | 7/13 | 오디오 I/O 모듈 / 1차 추론 파이프라인 완성 |
| 5 | 7/20 | 커스텀 데이터 전처리 / Dataset·DataLoader |
| 6 | 7/27 | Fine-tuning 루프 설계 / 모듈화 |
| 7 | 8/3  | Web UI 개발 / 백엔드 연동 |
| 8 | 8/7  | 통합 디버깅 / 문서화·최종 보고서 |
