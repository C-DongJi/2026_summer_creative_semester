# 음원 분리 파이프라인 (Music Source Separation Pipeline)

2026 하계 세종창의학기제(집중이수제) 프로젝트입니다.
보컬 제거(음원 분리) 모델을 VRAM이 작은 GPU에서도 안정적으로 돌릴 수 있는
파이프라인으로 만들고, 직접 모은 음원으로 파인튜닝까지 해보는 것이 목표입니다.

- 팀: 이준영, 안준석 (인공지능학과) / 지도교수: 신승협
- 모델 선정 이유와 전체 설계: [docs/PIPELINE_DESIGN.md](docs/PIPELINE_DESIGN.md)
- 주차별 진행 기록: [docs/PROGRESS.md](docs/PROGRESS.md)

## 프로젝트 소개

만들고자 하는 것은 크게 세 가지입니다.

1. **추론**: MP3/WAV를 넣으면 보컬과 반주(MR)를 분리해 줍니다. 곡 전체를 한 번에
   추론하면 GPU 메모리가 부족해지기 때문에, 8초 단위 청크로 잘라 추론한 뒤
   Overlap-Add(크로스페이드)로 이어 붙여서 OOM 없이 처리합니다.
2. **파인튜닝**: 직접 수집한 (원본, 반주) 음원 쌍만으로 모델을 재학습해서
   원하는 장르의 분리 성능을 끌어올립니다. 악기별로 분리된 데이터셋은 쓰지 않습니다.
3. **Web UI**: Gradio 화면에서 업로드, 분리, 다운로드를 코드 없이 할 수 있게 합니다.

모델은 2026년 7월 기준 리더보드를 조사해서 **Kim Mel-Band RoFormer** 체크포인트를
골랐습니다. 상위권 공개 모델 중 유일하게 MIT 라이선스이고, 8GB GPU에서 파인튜닝한
선례가 있습니다. 학습과 추론에는 [ZFTurbo의 MSST 프레임워크](https://github.com/ZFTurbo/Music-Source-Separation-Training)를
사용하고, 반주는 `mixture - vocals` 감산으로 복원합니다.

## 환경 설정

```bash
conda activate changuihakgi
pip install -r requirements.txt
bash scripts/setup_msst.sh    # MSST 클론 + Kim 체크포인트(약 913MB) 다운로드
```

GPU 작업(OOM 테스트, 파인튜닝)은 Colab 또는 GPU PC에서 진행합니다.
- Colab 연결: [docs/COLAB.md](docs/COLAB.md) / 검증 절차: [docs/GPU_SESSION1.md](docs/GPU_SESSION1.md)
- RTX 5060 Ti PC 세팅: [docs/SETUP_GPU_PC.md](docs/SETUP_GPU_PC.md) (50시리즈는 CUDA 12.8 빌드 필수)
- MoisesDB 장르 선택 재학습: [docs/FINETUNE_MOISESDB.md](docs/FINETUNE_MOISESDB.md)

## 사용법

```bash
# 보컬/반주 분리
python scripts/separate.py --input song.mp3 --output outputs/

# 베이스라인 OOM vs 청크 추론 비교 (GPU)
python scripts/verify_oom_chunking.py --minutes 4

# 수집한 (mix, inst) 쌍 -> 학습용 데이터셋 변환
python scripts/prepare_data.py --config config/default.yaml

# 공개 데이터셋(MoisesDB 등) 변환 + 장르 선택
python scripts/prepare_dataset.py --dataset moisesdb --src /data/moisesdb_v0.1 --genres rock pop
python scripts/check_dataset.py --dir data/processed   # 변환 결과 점검

# 파인튜닝 (자체 학습 루프)
python scripts/train.py --config config/default.yaml

# Web UI — 분리 + 사용자별 재학습 탭 (--host 0.0.0.0 이면 다른 기기에서 접속 가능)
python app/webui.py

# 테스트
python -m pytest tests/ -v
```

## 폴더 구조

```
├── config/            # 설정 (default.yaml, msst_finetune.yaml)
├── data/              # raw(수집 원본) / processed(학습셋) / valid(검증셋)
├── docs/              # 설계 문서, 진행 기록, 가이드
├── models/checkpoints/  # 사전학습·파인튜닝 가중치
├── src/
│   ├── audio/         # 오디오 I/O, 청크 분할 + Overlap-Add
│   ├── models/        # 모델 로딩 (MSST 경유)
│   ├── inference/     # 추론 파이프라인
│   ├── training/      # 전처리, Dataset, 손실 함수, 학습 루프
│   ├── data/          # 공개 데이터셋(MoisesDB/MedleyDB) 변환
│   └── utils/         # 설정 로딩, 디바이스 선택
├── scripts/           # 실행 스크립트
├── app/               # Gradio Web UI
├── tests/
└── third_party/       # MSST 체크아웃 (setup_msst.sh로 설치)
```

## 역할 분담

- **이준영**: 모델 조사·선정, 청크 분할 추론 + Overlap-Add 구현, 자체 학습 루프 작성
- **안준석**: 오디오 I/O와 전처리, Dataset/DataLoader, Web UI 및 백엔드 연동

## 진행 일정

| 주차 | 일시 | 내용 | 현황 |
|:---:|:---:|---|:---:|
| 1 | 6/26 | SOTA 논문 리뷰, 모델·프레임워크 선정 | 완료 |
| 2 | 6/29 | 개발 환경 세팅, 베이스라인 구동, OOM 한계 테스트 | 완료* |
| 3 | 7/6 | 청크 분할 추론 + 크로스페이드(Overlap-Add) 구현 | 완료 |
| 4 | 7/13 | 오디오 I/O 모듈, 1차 추론 파이프라인 완성 | 완료 |
| 5 | 7/20 | 커스텀 데이터 전처리, Dataset/DataLoader | 완료 |
| 6 | 7/27 | Fine-tuning 루프 설계 및 환경 모듈화 | 완료* |
| 7 | 8/3 | Web UI 개발, 백엔드 연동 | 진행 중 |
| 8 | 8/7 | 통합 디버깅, 문서화, 최종 보고서 | 예정 |

\* 2·6주차의 GPU 실측(OOM 측정, 실제 파인튜닝 실행)은 Colab GPU 세션에서 진행합니다.
자세한 내용은 [docs/PROGRESS.md](docs/PROGRESS.md) 참고.
