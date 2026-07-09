# 진행 계획표 (재정렬) — 2026-07-09 기준

재설계로 순서가 바뀌었고, GPU가 필요한 작업은 **클라우드/연구실 서버**에서 돌리기로 정리한 실행 계획.
원본 일정은 신청서 4항 참고. 마일스톤 날짜(7/13·7/20·7/27·8/3·8/7)는 유지하되 내용을 실제 진행 상태에 맞게 재배치.

## 실행 환경 분리 원칙

| 환경 | 용도 | 제약 |
|------|------|------|
| **로컬 (이 WSL, CPU 전용)** | 코드 개발·배선·전처리·Dataset·Web UI·짧은 클립 스모크 테스트 | torch CPU 빌드. VRAM OOM 재현 불가, 긴 곡 추론은 느림 |
| **클라우드/연구실 GPU (NVIDIA)** | 베이스라인 OOM 테스트, 실제 긴 곡 추론 검증, 파인튜닝, SDR 평가 | CUDA 빌드 torch 필요 (`--index-url .../cu121`), 세션 단위로 묶어서 진행 |

> GPU 서버에서는 로컬과 별도로 환경 재현 필요: `conda env create -f environment.yml` 후
> `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121` (CPU 빌드 대신 CUDA 빌드).

## 현재까지 완료된 것 (원 1~3주차 상당)

- ✅ **1주차 (SOTA 리뷰)** — 2026-07 리더보드·라이선스·프레임워크 재조사, Mel-Band RoFormer(Kim/MIT) + MSST 선정. → [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md)
- ✅ **3주차 (분할 추론 + 크로스페이드)** — 청크 분할 + Overlap-Add 알고리즘 **구현·단위테스트 완료**. → [src/audio/chunking.py](../src/audio/chunking.py), [tests/test_chunking.py](../tests/test_chunking.py)
  - ⚠️ 단, **항등 함수로만 검증**됨. 실제 모델 검증은 GPU 세션에서 필요.
- ✅ 부수 완료: conda 환경(`changuihakgi`), 전체 폴더/모듈 스캐폴딩, 전처리·Dataset·손실함수·학습 루프·추론 파이프라인 골격, multi-res STFT loss(+테스트)
- ⬜ **2주차 (베이스라인 구동 + OOM 테스트)** — **미완**. 실제 모델 다운로드·구동이 안 됨 → GPU 세션으로 이동

## 재정렬 일정

### ▶ ~7/13 : 로컬 배선 검증 + GPU 세션 ①
**로컬 (안준석 주도, 오디오 I/O)**
- [ ] 실제 MP3/WAV/FLAC + 다양한 샘플레이트 파일로 [io.py](../src/audio/io.py) 검증 (44.1kHz 스테레오 변환·리샘플 정확성)
- [ ] `bash scripts/setup_msst.sh` 실행 → MSST 클론 + Kim 체크포인트(~913MB) 다운로드 (로컬 저장은 CPU에서도 가능)
- [ ] **짧은 클립(5~10초)** 으로 [scripts/separate.py](../scripts/separate.py) 엔드투엔드 스모크 테스트 — 느려도 전체 배선(로드→청크→추론→감산→저장)이 도는지 확인

**GPU 세션 ① (이준영 주도) = 원 2주차 + 3주차 검증**
- [ ] 3~4분 실제 고음질 곡으로 **청크 없이** 추론 → VRAM 한계/OOM 관찰·기록 (2주차 OOM 테스트)
- [ ] **청크 분할 추론 켜고** 동일 곡 처리 → OOM 해소 확인 + 경계 잡음(틱) 없이 이어지는지 청취 검증 (3주차 알고리즘 실증)
- [ ] chunk_seconds·overlap 값 튜닝, VRAM 사용량 표 작성

### ▶ ~7/20 : 데이터 파이프라인 (원 5주차)
**로컬 (안준석 주도)**
- [ ] 실제 (mix, inst) 쌍 1~2곡 준비 → [scripts/prepare_data.py](../scripts/prepare_data.py) 실행
- [ ] 검증: `vocals = mix − inst` 감산 정확성, **공통 게인** 적용(개별 정규화 아님), 샘플 정렬, MSST 레이아웃(`<곡>/{vocals.wav, other.wav}`) 생성
- [ ] [Dataset/DataLoader](../src/training/dataset.py) 로 소량 데이터 로딩·세그먼트 추출 확인
- [ ] 운영자 수집 데이터셋 규모 확보 계획 (커뮤니티 기준 최소 ~40곡, 견고하려면 170곡+)

### ▶ ~7/27 : 파인튜닝 (원 6주차) — 주로 GPU
**GPU 세션 ② (이준영 주도)**
- [ ] 경로 A: MSST `train.py` 로 Kim 체크포인트 파인튜닝 (검증된 기본 경로) — 명령어는 [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md) §3
- [ ] 경로 B: 자체 학습 루프 [src/training/train.py](../src/training/train.py) 구동 (창의학기제 학습 목표)
- [ ] 8/16GB VRAM 프리셋 적용 (chunk·batch·grad-accum·AMP·adamw8bit), 필요시 LoRA
- [ ] 홀드아웃 ≥5곡 SDR 매 epoch 추적 (무증상 품질저하 방지)

### ▶ ~8/3 : Web UI (원 7주차)
**로컬 (안준석 주도)**
- [ ] [app/webui.py](../app/webui.py) Gradio 레이아웃 완성 (업로드 → 분리 → vocal/inst 다운로드)
- [ ] 백엔드 추론 연동 (짧은 클립은 로컬 CPU, 긴 곡·데모는 GPU 서버 연결)
- [ ] 파인튜닝된 가중치를 UI에서 선택·로드하는 경로

### ▶ ~8/7 : 통합·문서화 (원 8주차)
- [ ] [전처리 → 추론 → 후처리] + 재학습 전체 엣지케이스 디버깅
- [ ] GitHub README·설계 문서 정리, 최종 결과 보고서 작성
- [ ] (선택) unwa Leap 등 다른 공개 모델과 추론 품질 A/B 비교

## 지금 당장 할 일 (요약)

**로컬에서 바로 착수 가능** (GPU 불필요): 오디오 I/O 실제 파일 검증 → `setup_msst.sh`로 모델 확보 → 짧은 클립 스모크 테스트.
**GPU 서버 잡히면 묶어서**: 베이스라인 OOM 테스트(2주차) + 청크 로직 실증(3주차 검증) → 이후 파인튜닝(6주차).
