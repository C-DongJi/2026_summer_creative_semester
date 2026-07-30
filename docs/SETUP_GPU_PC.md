# GPU PC(RTX 5060 Ti) 세팅 가이드

다른 컴퓨터(RTX 5060 Ti 장착)에서 이 프로젝트를 돌리기 위한 절차.
5060 Ti는 8GB/16GB 두 버전이 있다 — 아래 VRAM 프리셋 표에서 자기 버전을 확인할 것.

## ⚠️ 가장 중요한 것: PyTorch CUDA 버전

RTX 50 시리즈(Blackwell, sm_120)는 **CUDA 12.8 이상 빌드의 PyTorch가 필수**다.
구버전 휠(cu121 등)을 설치하면 `sm_120 is not compatible` 오류와 함께 GPU를 못 쓴다.

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

(다른 문서에 나오는 cu121 명령은 구형 GPU 기준 — 5060 Ti에서는 반드시 cu128)

## 1. 사전 준비 (Windows)

1. **NVIDIA 드라이버** 최신 설치 (RTX 50 지원 버전) — 설치 후 PowerShell에서
   `nvidia-smi`가 5060 Ti를 표시하는지 확인
2. **WSL2 + Ubuntu** 설치 권장 (프로젝트가 WSL 기준으로 개발됨):
   ```powershell
   wsl --install -d Ubuntu
   ```
   Windows 드라이버만 깔면 WSL 안에서 GPU가 자동 인식된다 (WSL 전용 CUDA 툴킷
   설치 불필요 — 드라이버가 제공). WSL 터미널에서 `nvidia-smi` 확인.
3. WSL 안에 **Miniconda** 설치:
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

## 2. 프로젝트 설치

```bash
# 코드
git clone -b leejy https://github.com/C-DongJi/2026_summer_creative_semester.git
cd 2026_summer_creative_semester

# 환경 (Python 3.11 + ffmpeg)
conda create -y -n changuihakgi python=3.11
conda activate changuihakgi
conda install -y -c conda-forge ffmpeg

# ★ 5060 Ti용 CUDA 빌드 torch (cu128!)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 나머지 의존성 + MSST + Kim 체크포인트(~913MB)
pip install soundfile librosa einops tqdm pyyaml gradio
bash scripts/setup_msst.sh
```

## 3. 동작 확인

```bash
# GPU 인식 (True + 5060 Ti가 떠야 함)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 테스트 스위트
python -m pytest tests/ -q

# 베이스라인 OOM vs 청크 추론 (2·3주차 검증 겸 GPU 동작 확인)
python scripts/verify_oom_chunking.py --minutes 4
```

## 4. VRAM 프리셋

| 항목 | 5060 Ti 8GB | 5060 Ti 16GB |
|---|---|---|
| 추론 | 기본 설정 그대로 (fp16 자동) | 기본 + `inference.batch_size: 2~4` |
| 학습 chunk_size | 131584 (기본값 유지) | 352800으로 상향 가능 |
| 학습 옵션 | adamw8bit + grad accum 8 | adam + grad accum 4~8 |

- 추론은 `inference.precision: auto`가 반정밀(fp16/bf16)을 자동 적용해 VRAM을 절반으로 씀
- 학습 상세는 [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md) §3, MoisesDB 장르 재학습은
  [FINETUNE_MOISESDB.md](FINETUNE_MOISESDB.md) 참고

## 5. Web UI 실행

```bash
# 이 PC에서만 사용
python app/webui.py

# 같은 공유기/네트워크의 다른 기기(노트북·폰)에서 접속 허용
python app/webui.py --host 0.0.0.0
# -> 다른 기기 브라우저에서 http://<이 PC의 IP>:7860
#    (Windows 방화벽에서 7860 인바운드 허용 필요할 수 있음)

# 외부 임시 공개 링크 (gradio 터널)
python app/webui.py --share
```

GPU가 있으므로 분리도 빠르고(곡당 수 초~수십 초), **재학습 탭의 사용자별
파인튜닝도 이 PC에서 실용적으로 돈다** (CPU에서는 데모 수준만 가능).

## 문제 해결

- `sm_120 is not compatible with the current PyTorch installation` → torch가 cu128이
  아님. `pip uninstall torch torchaudio` 후 2단계의 cu128 명령으로 재설치
- WSL에서 `nvidia-smi` 없음 → Windows 드라이버가 구버전이거나 WSL 재시작 필요
  (`wsl --shutdown` 후 재진입)
- 체크포인트 다운로드 중단 → `bash scripts/setup_msst.sh` 재실행 (이어받기 지원)
