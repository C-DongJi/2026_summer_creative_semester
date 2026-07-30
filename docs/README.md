# 문서 안내

문서가 늘어나서 용도별로 정리한 지도. 처음 보는 사람은 아래 순서대로 읽으면 된다.

1. 루트 [README](../README.md) - 프로젝트 개요, 사용법, 일정
2. [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md) - 모델 선정 이유와 파이프라인 구조
3. 환경에 맞는 설치 문서 - 5060 Ti PC면 [SETUP_GPU_PC.md](SETUP_GPU_PC.md)
4. [TESTING.md](TESTING.md) - 설치 후 기능 테스트
5. [EXPERIMENT.md](EXPERIMENT.md) - 재학습 실험 (데이터 준비부터 보고서까지)

## 목적별로 찾기

| 하려는 일 | 볼 문서 |
|---|---|
| 왜 이 모델/프레임워크인지, 전체 구조 | [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md) |
| 주차별로 뭘 했는지 (산출물·검증 기록) | [PROGRESS.md](PROGRESS.md) |
| 남은 작업, GPU 세션 체크리스트 | [SCHEDULE.md](SCHEDULE.md) |
| 5060 Ti PC 처음부터 세팅 | [SETUP_GPU_PC.md](SETUP_GPU_PC.md) |
| 세팅 끝난 PC에서 기능 테스트 | [TESTING.md](TESTING.md) |
| 데이터셋 후보·라이선스·2-stem 변환 원리 | [DATASETS.md](DATASETS.md) |
| MoisesDB 받아서 장르별로 준비 (80GB 대응 포함) | [FINETUNE_MOISESDB.md](FINETUNE_MOISESDB.md) |
| 재학습 실험 실행 절차 | [EXPERIMENT.md](EXPERIMENT.md) |
| 평가 설계와 판정 기준의 근거 | [EVALUATION.md](EVALUATION.md) |
| Colab GPU 연결 (대안 환경) | [COLAB.md](COLAB.md), [GPU_SESSION1.md](GPU_SESSION1.md) |

## 문서 사이 관계

- 실험 **실행 절차**는 [EXPERIMENT.md](EXPERIMENT.md)가 기준이고,
  [EVALUATION.md](EVALUATION.md)는 그 **설계 근거**(평가셋 구성, 누설 방지, 판정
  기준의 출처)를 설명한다. 두 문서의 명령이 다르면 EXPERIMENT.md를 따른다.
- GPU 작업의 기본 환경은 **5060 Ti PC**다. [COLAB.md](COLAB.md)와
  [GPU_SESSION1.md](GPU_SESSION1.md)는 Colab T4로 할 때의 대안 절차로 남겨둔 것.
- 진행 **기록**은 [PROGRESS.md](PROGRESS.md), 남은 일 관리는 [SCHEDULE.md](SCHEDULE.md).
- 판정 기준표는 [EVALUATION.md](EVALUATION.md) 5절이 원본이고,
  `scripts/make_report.py`가 같은 기준을 코드로 구현해 보고서에 자동 적용한다.
