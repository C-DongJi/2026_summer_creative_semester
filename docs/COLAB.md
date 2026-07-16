# Colab GPU를 VS Code에서 구동하기 (2·3주차 검증용)

로컬 WSL은 CPU 전용이라 VRAM OOM을 재현할 수 없다. Colab의 무료 GPU(T4 16GB 등)를
빌리되, 익숙한 VS Code에서 코드를 편집·실행하기 위해 **VS Code Remote Tunnel**로 연결한다.

> 핵심 개념: Colab은 일반 SSH 서버를 안 열어준다. 대신 Colab VM 안에서
> `code tunnel`(VS Code 공식 터널)을 띄우고, 로컬 VS Code가 그 터널에 붙는다.
> 그러면 **VS Code는 로컬 화면, 실제 실행은 Colab GPU** 위에서 일어난다.

---

## 0. 사전 준비 (한 번만)

- 로컬 VS Code에 **"Remote - Tunnels"** 확장 설치 (Microsoft 제공)
- GitHub 계정 (터널 인증에 사용) — Microsoft 계정도 가능
- **코드를 GitHub에 올려두기**: Remote Tunnel은 Colab의 파일시스템에 붙으므로,
  로컬 파일이 자동으로 가지 않는다. Colab에서 `git clone` 하려면 재설계 커밋이
  원격에 있어야 한다.
  - 아직 push가 안 된 상태라면 먼저 push (GitHub 인증 필요).
  - push가 막혀 있으면 → 아래 **부록 B(구글 드라이브 업로드)** 로 우회.

---

## 1. Colab 노트북에서 GPU 런타임 + 터널 실행

새 Colab 노트북을 열고:

**① GPU 켜기**: 메뉴 `런타임 → 런타임 유형 변경 → 하드웨어 가속기: T4 GPU` → 저장

**② GPU 확인** (셀):
```python
!nvidia-smi
import torch; print("CUDA:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
```

**③ 터널을 셀에서 백그라운드로 실행** (권장 — Colab 터미널 불필요):

> Colab 터미널(xterm)은 붙여넣기가 깨지는 경우가 많다(bracketed paste 문제).
> 터널은 셀에서 nohup 백그라운드로 띄우고, 이후 작업은 전부
> 접속된 VS Code의 통합 터미널(붙여넣기 정상)에서 하는 것이 안정적이다.

```python
# 셀 1: 터널 백그라운드 실행 (셀이 점유되지 않음)
!wget -q https://raw.githubusercontent.com/C-DongJi/2026_summer_creative_semester/leejy/scripts/colab_tunnel.sh -O /content/tunnel.sh
!nohup bash /content/tunnel.sh > /content/tunnel.log 2>&1 &
```
```python
# 셀 2: 몇 초 뒤 실행 — 인증 코드 확인
import time; time.sleep(8)
!tail -20 /content/tunnel.log
```
- 출력의 `github.com/login/device` 주소에서 8자리 코드 입력(GitHub 인증).
- 셀 2를 다시 실행해 `vscode.dev/tunnel/colab-gpu/...` 링크가 보이면 터널 가동 중.
- 터널은 백그라운드 프로세스라 셀은 자유롭게 쓸 수 있고, 브라우저 탭은 열어둘 것(세션 유지).

<details><summary>대안: Colab 터미널(Pro 터미널/colab-xterm)에서 직접 실행</summary>

터미널 붙여넣기가 정상인 환경이면 아래 한 줄로도 된다 (`!` 없이 — `!`는 셀 전용):
```bash
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' -o vscode_cli.tar.gz \
  && tar -xf vscode_cli.tar.gz \
  && ./code tunnel --accept-server-license-terms --name colab-gpu
```
붙여넣기가 `^[[200~` 등으로 깨지면 `bind 'set enable-bracketed-paste on'` 입력 후 재시도.
</details>

---

## 2. 로컬 VS Code에서 터널에 접속

- `Ctrl+Shift+P` → **"Remote-Tunnels: Connect to Tunnel"** → GitHub 계정 선택 →
  `colab-gpu` 선택.
- 또는 Colab이 출력한 `vscode.dev/tunnel/...` 링크를 브라우저에서 열어도 된다.
- 접속되면 좌하단에 `Tunnel: colab-gpu` 표시. 이제 VS Code 터미널·편집기가
  **Colab VM 위**에서 동작한다.

---

## 3. Colab 위에서 코드·환경 세팅 (VS Code 터미널)

접속된 VS Code에서 새 터미널을 열고 (여기서 실행되는 명령은 전부 Colab GPU 머신에서 돈다):

```bash
# ① 코드 가져오기 (public repo clone은 인증 불필요)
git clone https://github.com/C-DongJi/2026_summer_creative_semester.git
cd 2026_summer_creative_semester

# ② 의존성: Colab에는 CUDA 빌드 torch가 이미 있음 → torch 재설치 금지.
#    나머지 라이브러리만 설치.
pip install soundfile librosa einops tqdm pyyaml

# ③ MSST 프레임워크 + Kim 체크포인트(~913MB) 다운로드
bash scripts/setup_msst.sh
```

> ⚠️ `requirements.txt`를 통째로 `pip install` 하면 Colab의 CUDA torch를
> CPU 버전으로 덮어쓸 수 있다. **torch는 건드리지 말고** 위처럼 개별 설치할 것.

---

## 4. 2·3주차 검증 실행

```bash
# 합성 4분 노이즈로 베이스라인 vs 청크 VRAM 비교
python scripts/verify_oom_chunking.py --minutes 4

# 베이스라인 OOM을 확실히 유도하려면 길이를 늘린다 (T4 16GB 기준 6~10분)
python scripts/verify_oom_chunking.py --minutes 8

# 실제 곡으로 검증 (파일을 Colab에 업로드 후)
python scripts/verify_oom_chunking.py --input outputs/song.mp3
```

기대 출력: 베이스라인은 길이에 따라 peak VRAM이 커지다 결국 `OOM`,
청크 방식은 `ok`로 낮은 VRAM 고정 → **3주차 알고리즘 효과 실증**.

실제 분리 결과물까지 들어보려면:
```bash
python scripts/separate.py --input outputs/song.mp3 --output outputs/
```

측정한 VRAM 표·소요 시간을 [SCHEDULE.md](SCHEDULE.md)의 GPU 세션 ① 체크리스트에 기록.

---

## Colab 사용 시 주의

- **세션 제한**: 무료 티어는 유휴 ~90분/최대 ~12시간 후 끊김. 끊기면 3장부터 다시.
- **디스크 초기화**: 런타임이 끊기면 clone·다운로드한 것이 사라진다. 체크포인트를
  구글 드라이브에 저장해두면 재다운로드를 아낄 수 있다 (부록 A).
- **터널 프로세스**: 백그라운드 터널이 죽으면 연결이 끊긴다.
  상태 확인: 셀에서 `!tail -5 /content/tunnel.log` / 재시작: 셀 1 다시 실행.

---

## 부록 A — 체크포인트를 드라이브에 캐시 (재다운로드 절약)

```python
from google.colab import drive
drive.mount('/content/drive')
```
드라이브에 한 번 받아두고, 이후 세션에선 복사만:
```bash
mkdir -p models/checkpoints
cp /content/drive/MyDrive/msst/MelBandRoformer.ckpt models/checkpoints/
```

## 부록 B — push가 막혔을 때: 드라이브로 코드 전달

로컬에서 repo를 zip으로 묶어 구글 드라이브에 올린 뒤, Colab에서:
```python
from google.colab import drive; drive.mount('/content/drive')
```
```bash
unzip /content/drive/MyDrive/creative_semester.zip -d ~/proj
cd ~/proj/creative_semester
```
그 뒤 3장 ②부터 동일.

## 부록 C — 대안: SSH(cloudflared)로 Remote-SSH 접속

`code tunnel` 대신 SSH를 선호하면 `colab_ssh` + cloudflared로 Remote-SSH도 가능하지만,
`code tunnel`이 계정 설정이 더 간단해 기본 권장. (필요 시 별도 안내)
