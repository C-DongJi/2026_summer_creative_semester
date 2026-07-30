# 주차별 진행 기록

> 신청서(2026-05) 계획 일정 기준. 각 주차의 목표 대비 산출물과 검증 결과를 기록한다.

## 1주차 (6/26) — SOTA 동향 파악 및 논문 심층 리뷰 ✅

- HTDemucs, BS-RoFormer, Mel-RoFormer 논문 리딩 및 MVSep multisong 리더보드 분석
  → 상위 10개 전부 RoFormer 계열, HTDemucs는 3~4dB 열세 + 리포 아카이브 확인
- 리더보드·라이선스·파인튜닝 선례를 종합해 **Kim Mel-Band RoFormer(MIT) + ZFTurbo MSST
  프레임워크**로 모델 확정 — 상위권 공개 모델 중 유일한 MIT, 8GB 파인튜닝 선례 보유
- 핵심 근거 9건을 개별 웹 검증으로 확인
- 산출물: [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md) (선정 근거·하이퍼파라미터·함정 목록)

## 2주차 (6/29) — 개발 환경 세팅 및 베이스라인 구동 테스트 ✅

- conda 가상환경 `changuihakgi`(Python 3.11) 구축, 전체 폴더/모듈 스캐폴딩
- [scripts/setup_msst.sh](../scripts/setup_msst.sh): MSST 클론 + Kim 체크포인트(913MB) 자동 설치
  (torch 빌드 보존·GUI 패키지 제외·setuptools 고정 이슈 해결 포함)
- [scripts/verify_oom_chunking.py](../scripts/verify_oom_chunking.py): 베이스라인(일괄 추론) vs
  청크 추론의 피크 VRAM·시간 비교 스크립트 — RoFormer 어텐션 메모리가 길이 제곱으로
  증가함을 근거로 T4(16GB)에서 3~4분 곡 OOM 예상 재원 산정
- Colab GPU 연결 환경 구축: [COLAB.md](COLAB.md) (VS Code Remote Tunnel, 백그라운드 터널)
- OOM **실측 완료** (5060 Ti PC): 4분 곡 fp32 일괄 추론이 23.63GB를 요구하며 OOM —
  "긴 곡 일괄 추론은 소비자용 GPU에서 불가"를 실증. 길이별 상세 측정표는
  [SCHEDULE.md](SCHEDULE.md) GPU 세션 ① 체크리스트에서 마무리 중

## 3주차 (7/6) — 분할 추론 아키텍처 설계 (청크 + 크로스페이드) ✅

- [src/audio/chunking.py](../src/audio/chunking.py): 청크 분할 추론 + Overlap-Add 직접 구현
  - Hann 윈도우 크로스페이드 + 가중치 정규화로 경계선 잡음(틱) 제거
  - 반사 패딩으로 곡 시작/끝 묵음 버그 해결, 청크보다 짧은 입력 처리
  - 파라미터를 MSST demix() 규격에 정합: 8초 청크(352,800샘플), step = chunk/4 (75% 오버랩)
- 검증: 항등 함수 재구성 단위테스트 — 병합 결과가 원본과 일치(오차 < 1e-4)

## 4주차 (7/13) — 오디오 I/O 모듈 및 1차 추론 파이프라인 완성 ✅

- [src/audio/io.py](../src/audio/io.py): soundfile 기반 I/O — MP3/WAV/FLAC/OGG, 임의
  샘플레이트/채널 → 44.1kHz 스테레오 변환 (torchaudio TorchCodec 의존성 문제 회피)
  - 검증: [scripts/verify_audio_io.py](../scripts/verify_audio_io.py) 8/8 통과 (float32 roundtrip 포함)
- [src/models/registry.py](../src/models/registry.py): MSST 경유 Kim 체크포인트 로딩 —
  실모델(228M 파라미터) 로드 및 forward 규격 확인
- [src/inference/pipeline.py](../src/inference/pipeline.py) + [scripts/separate.py](../scripts/separate.py):
  [로드→청크→추론→Overlap-Add→감산 복원(inst = mix − vocals)→저장] 1차 파이프라인 완성
  - E2E 검증: 합성 곡 분리 후 vocals + instrumental vs mixture 최대 오차 **6.1e-05**

## 5주차 (7/20) — 커스텀 데이터 재학습 전처리 환경 구축 ✅

- [scripts/prepare_data.py](../scripts/prepare_data.py) / [src/training/preprocess.py](../src/training/preprocess.py):
  (mix, inst) 쌍 → MSST dataset_type 4 레이아웃(vocals/other) 자동 변환
  - **공통 게인** 정규화(개별 정규화 시 감산 관계가 깨지는 문제 해결), 샘플 정렬 검증
    (0.5초 초과 불일치 시 스킵), vocals = mix − inst float32 감산, 32-bit float 저장
- [scripts/check_dataset.py](../scripts/check_dataset.py): 학습셋 자동 점검 — 스템 누락/길이·SR·채널
  불일치/NaN/무음 보컬/클리핑 검출
- [src/training/dataset.py](../src/training/dataset.py): PyTorch Dataset/DataLoader (랜덤 세그먼트,
  mixture = vocals + other 재구성)
- 공개 데이터셋 인제스트: [scripts/prepare_dataset.py](../scripts/prepare_dataset.py) —
  MoisesDB/MedleyDB를 2-stem으로 접어서 활용 ([DATASETS.md](DATASETS.md))
- E2E 검증: 감산 정합성 오차 2.98e-08, 배치 내 mixture == vocals + other 성립,
  불량 쌍(길이 불일치) 자동 스킵 확인

## 6주차 (7/27) — Fine-tuning 루프 설계 및 환경 모듈화 🔺

- **경로 A (MSST, 기본)**: [config/msst_finetune.yaml](../config/msst_finetune.yaml) — Kim 원본
  레시피 파생(모델 섹션 체크포인트 호환 유지), T4/8GB VRAM 프리셋(batch 1 + grad accum 8,
  lr 1e-5, patience 1000, AMP), 실행 명령 주석 포함, LoRA 폴백(r=8, alpha=16) 문서화
- **경로 B (자체 학습 루프, 창의학기제 학습 목표)**: [src/training/train.py](../src/training/train.py)
  — 단일 타깃(vocals) 학습, AMP + gradient accumulation, 체크포인트 저장
- [src/training/losses.py](../src/training/losses.py): 체크포인트 원 손실인 multi-resolution STFT
  loss 직접 구현 (windows [4096,2048,1024,512,256], hop 147)
- 검증: 더미 모델 주입 dry-run 테스트 통과 (루프 완주·체크포인트 저장·multi-STFT 경로·가중치 업데이트)
- 🔺 잔여: 실데이터 확보 후 **실학습 실행**은 GPU 세션 ②에서 수행

## 7주차 (8/3) — Web UI 개발 및 백엔드 연동 🔄 진행 중

- [app/webui.py](../app/webui.py): Gradio 2탭 구성 완성
  - **분리 탭**: 업로드 → 보컬/반주 분리 → 재생·다운로드. 모델 드롭다운(기본/파인튜닝
    모델 선택), 작업별 소요 시간 표시, 5060 Ti PC에서 실곡 분리 동작 확인
  - **재학습 탭**: 사용자별로 (mix, inst) 쌍 업로드 → 데이터셋 축적 → 에폭/lr 지정
    재학습 → 완료 시 분리 탭 드롭다운에 자동 등록
- 실험·평가 도구 완비: [scripts/evaluate.py](../scripts/evaluate.py)(다중 모델·평가셋
  SDR 비교), [scripts/make_report.py](../scripts/make_report.py)(판정 기준 자동 적용
  보고서 생성), [scripts/extract_moisesdb_subset.py](../scripts/extract_moisesdb_subset.py)
  (80GB zip에서 장르 선별 추출)
- 잔여: 재학습 실험 실행([EXPERIMENT.md](EXPERIMENT.md)), 실사용 피드백 반영

## 요약

| 주차 | 계획일 | 상태 |
|---|---|---|
| 1 | 6/26 | ✅ 완료 (모델·프레임워크 확정) |
| 2 | 6/29 | ✅ 완료 (환경 세팅 + OOM 실측) |
| 3 | 7/6 | ✅ 완료 (Overlap-Add 구현·검증) |
| 4 | 7/13 | ✅ 완료 (1차 추론 파이프라인 E2E) |
| 5 | 7/20 | ✅ 완료 (전처리·Dataset·점검 도구) |
| 6 | 7/27 | 🔺 환경 완비, 실학습만 GPU 대기 |
| 7 | 8/3 | 🔄 진행 중 — Web UI 구현 완료, 실험 도구 완비 |
| 8 | 8/7 | 예정 — 통합 디버깅·문서화·최종 보고서 |

테스트 스위트: 39/39 통과 (청크 7 · 손실 4 · 인제스트 6 · 학습 루프 4 · 지표 5 ·
웹 UI 헬퍼 5 · 디바이스 4 · 보고서 4)
