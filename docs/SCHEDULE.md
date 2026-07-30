# 실행 계획표 - 2026-07-30 기준

남은 작업과 GPU 세션 체크리스트. 주차별로 이미 끝난 산출물의 기록은
[PROGRESS.md](PROGRESS.md)에 있고, 여기서는 앞으로 할 일만 관리한다.
원본 일정은 신청서 4항 참고. 마일스톤 날짜(8/3, 8/7)는 유지한다.

## 실행 환경

| 환경 | 용도 | 비고 |
|------|------|------|
| **로컬 WSL (CPU 전용)** | 코드 개발, 전처리, 단위 테스트, 짧은 클립 스모크 | torch CPU 빌드. OOM 재현 불가 |
| **5060 Ti PC (기본 GPU 환경)** | OOM 실측, 파인튜닝, 평가, Web UI 시연 | **cu128 빌드 필수** (sm_120). 설치: [SETUP_GPU_PC.md](SETUP_GPU_PC.md) |
| Colab T4 (대안) | GPU PC를 못 쓸 때 백업 | torch 프리설치(재설치 금지). 연결: [COLAB.md](COLAB.md) |

## 완료된 것 (원 1~6주차 + 7주차 일부)

- 1~6주차 산출물 전체: 모델 선정, 청크 분할 + Overlap-Add, 추론 파이프라인,
  전처리/Dataset, 학습 루프와 설정 - 상세는 [PROGRESS.md](PROGRESS.md)
- Web UI(원 7주차): 분리 탭 + 사용자별 재학습 탭 구현, 5060 Ti에서 실곡 분리 동작 확인
- 실험 도구: `scripts/evaluate.py`(다중 모델/평가셋 SDR 비교),
  `scripts/make_report.py`(보고서 자동 생성), `scripts/extract_moisesdb_subset.py`
  (80GB zip에서 장르만 선별 추출)
- **GPU 세션 ① 일부**: 4분 곡 fp32 일괄 추론이 23.63GB를 요구하며 **OOM 실증**
  (2주차 목표 달성). 청크 추론은 웹 UI로 실곡 분리 정상 동작 확인

## 남은 일

### GPU 세션 ① 마무리 - 측정표 완성

- [ ] fp16 수정이 반영된 최신 코드(`git pull`)로 재측정:
      `python scripts/verify_oom_chunking.py --minutes 1` (2, 4, 8도 반복)
- [ ] 길이별 [베이스라인 vs 청크] 상태·피크 VRAM 표 기록 - 표 양식은
      [TESTING.md](TESTING.md) 2단계

### GPU 세션 ② - 재학습 실험 (원 6주차 실측 + 결과 도출)

- [ ] MoisesDB 80GB 다운로드 (D드라이브) 후 장르 분포 확인:
      `python scripts/extract_moisesdb_subset.py --zip /mnt/d/... --list-genres`
- [ ] [EXPERIMENT.md](EXPERIMENT.md) 절차대로 진행:
      팝 추출·변환 → 점검 → 파인튜닝 → `evaluate.py` (base vs ft, 팝/타 장르)
      → `make_report.py` 보고서 생성
- [ ] (경로 B) 자체 학습 루프 `src/training/train.py`도 실데이터로 1회 구동
      (창의학기제 학습 목표 증빙)

### ~8/3 : Web UI 마무리 (원 7주차)

- [x] Gradio 분리 탭 (모델 드롭다운, 소요 시간 표시, 결과 다운로드)
- [x] 재학습 탭 (사용자별 데이터셋 업로드, 진행률, 완료 시 드롭다운 자동 등록)
- [ ] 실사용 중 나오는 불편/버그 반영

### ~8/7 : 통합·문서화 (원 8주차)

- [ ] 전체 파이프라인 엣지케이스 디버깅 (긴 곡, 모노 입력, 짧은 클립 등)
- [ ] 최종 보고서 작성: `make_report.py` 출력 + GPU 측정표 + 청취 평가 종합
- [ ] (선택) unwa Leap 등 다른 공개 모델과 추론 품질 A/B 비교
