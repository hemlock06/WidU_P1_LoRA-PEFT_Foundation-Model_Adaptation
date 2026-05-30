# fairseq-signals LoRA 패치

`fairseq-signals/`는 용량 문제로 git 추적에서 제외(.gitignore)되지만, 본 프로젝트에서 직접 구현한 **LoRA 코드는 이 라이브러리 내부를 수정**한다. 그 수정분을 이 패치로 보존한다.

## 포함 내용 (`lora_fairseq_signals.patch`)

- `fairseq_signals/modules/multi_head_attention.py` — `MultiHeadAttention`에 LoRA 주입(`enable_lora`, lora_A/lora_B, forward 분기)
- `fairseq_signals/models/classification/ecg_transformer_classifier.py` — LoRA config 필드 + 사후 주입/freezing
- `setup.py` — py39 설치 대응

## 재현 (적용 방법)

```bash
# 1. base commit으로 fairseq-signals clone
git clone https://github.com/Jwoo5/fairseq-signals.git
cd fairseq-signals
git checkout f8f0ff1c788a82c2059cb452cd5462898867489e

# 2. 패치 적용
git apply ../patches/lora_fairseq_signals.patch

# 3. py39 env에 editable 설치
pip install --editable ./
```

base commit은 `BASE_COMMIT.txt` 참조.
