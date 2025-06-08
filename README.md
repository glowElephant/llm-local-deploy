# llm-local-deploy

이 저장소에는 **`run_llm.py`** 스크립트만 Git으로 관리합니다.  
모든 다른 파일은 `.gitignore` 설정을 통해 무시 처리하여, `run_llm.py`만 추적됩니다.

---

## 사전 준비

- **운영체제**: Windows 10 또는 Windows 11  
- **Python**: 3.11 (https://www.python.org/downloads/windows/ 에서 설치, “Add Python to PATH” 옵션 체크)  
- **(선택) GPU**: NVIDIA RTX 4060 이상 + CUDA 12.5 드라이버 설치  
- **Hugging Face 계정 및 토큰**:  
  - https://huggingface.co/settings/tokens 에서 **Read** 권한 토큰 생성  
  - 이후 `huggingface-cli login --token <YOUR_TOKEN>` 수행

---

## 1. 가상환경 생성 및 활성화

CMD(또는 PowerShell)를 열고, 프로젝트 디렉토리에서 다음을 실행하세요:

```bat
# 1) 가상환경 생성
python -m venv llm-env

# 2) 가상환경 활성화
#  - CMD:
llm-env\Scripts\activate.bat

#  - PowerShell:
llm-env\Scripts\Activate.ps1
```

프롬프트에 `(llm-env)`가 표시되면 활성화된 상태입니다.

---

## 2. 라이브러리 설치

### A) GPU 지원 (Conda + pip 병행 권장)

```bat
# (conda가 설치되어 있어야 합니다)
conda create -n llm-env python=3.11 -y
conda activate llm-env

# PyTorch + CUDA 12.5
conda install pytorch pytorch-cuda=12.5 -c pytorch -c nvidia -y

# Hugging Face, Accelerate, SentencePiece
pip install transformers accelerate sentencepiece

# bitsandbytes GPU 빌드 (CUDA12.5용)
pip install bitsandbytes-cuda125
```

> 설치 후, `python - <<EOF
import torch, bitsandbytes as bnb; print(torch.cuda.is_available(), bnb.available_backends())
EOF`  
> 를 실행해 `True` 및 `{'cuda', ...}` 출력 확인

### B) CPU 전용 (테스트용)

```bat
(llm-env) C:\Git\llm-local-deploy> python -m pip install --upgrade pip
(llm-env) C:\Git\llm-local-deploy> python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu125 --trusted-host download.pytorch.org
(llm-env) C:\Git\llm-local-deploy> pip install transformers accelerate sentencepiece
# bitsandbytes는 제거하여 사용하지 않습니다.
(llm-env) C:\Git\llm-local-deploy> pip uninstall -y bitsandbytes
```

---

## 3. Hugging Face 로그인

```bat
(llm-env) C:\Git\llm-local-deploy> huggingface-cli login --token <YOUR_TOKEN>
```

성공 시 `Login successful` 메시지가 출력됩니다.

---

## 4. `.gitignore` 설정

```gitignore
*
!run_llm.py
```

위 내용을 프로젝트 루트의 `.gitignore`에 추가하세요.  
모두 무시(`*`)하고 `run_llm.py`만 추적(`!run_llm.py`)합니다.

---

## 5. 실행 방법

가상환경이 활성화된 상태에서:

```bat
# 기본 실행
(llm-env) C:\Git\llm-local-deploy> python run_llm.py

# Accelerate 설정 후(멀티 GPU 또는 mixed-precision)
(llm-env) C:\Git\llm-local-deploy> accelerate launch run_llm.py
```

실행 후, 프롬프트가 나타나면 메시지를 입력하고 **Enter**:

```
[You] 안녕하세요!
[LLM] 안녕하세요! 무엇을 도와드릴까요?
```

`exit` 또는 `quit` 입력 시 종료합니다.

---

## 6. 모델 변경하기

`run_llm.py` 상단의 `model_id`만 바꾸면 됩니다. 예:

```diff
- model_id = "meta-llama/Llama-2-7b-chat-hf"
+ model_id = "tiiuae/falcon-7b-instruct"
```

다른 모델 ID로 바꾼 뒤 저장하고 재실행하세요.
