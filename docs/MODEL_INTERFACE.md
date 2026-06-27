# MODEL_INTERFACE — P1 추론 입출력 계약 (P2 연결 규약)

> P1을 다운스트림(규칙 결합기 / P2 멀티모달 융합 / 로깅)으로 통합할 때의 입출력 계약.
> **명세 문서가 규정한 계약(spec)** 과 **실제 코드가 구현한 출력(as-built)** 을 분리해 기술한다.
> 둘 사이에 드리프트가 있으며(§4), 이는 P2 연결 전 반드시 확인해야 한다.
>
> 1차 출처(코드): `scripts/p1_cardiac_channel.py`, `scripts/p1_cardiac_logging_adapter.py`.
> 1차 출처(명세): `records/00_research_plan.md` §5, `README.md` "P1 → P2 인터페이스".

---

## 1. 입력 계약 (검증됨)

`P1CardiacChannel.infer(signal, batch_size=32)` — `signal`:

| 항목 | 규격 |
|---|---|
| shape | `(12, 5000)` 단일 또는 `(N, 12, 5000)` 배치 |
| dtype | float32 (내부에서 `np.asarray(..., float32)` 캐스팅) |
| 샘플링 | 500Hz, 10초 = 5000샘플 |
| 진폭 | **raw mV, 정규화 없음** (ECG-FM 사전학습 `normalize=False`와 일치) |
| lead 순서 | 표준 12-lead 슬롯 |
| N-lead 입력 | 부족 lead는 **0-fill**. 단일리드는 **slot 1 (Lead II)** 에 싣고 나머지 0 |

- 반환 형태: 입력이 `(12,5000)`이면 스칼라/벡터, `(N,12,5000)`이면 배열(`N`축 유지).
- 단일리드 0-fill 근거: ECG-FM RLM 사전학습으로 0-fill = lead 마스킹과 동일하게 처리됨
  (N-lead ablation에서 1-lead까지 AUROC 유지로 검증, `records/03`).

`CardiacLoggingAdapter.to_record(ecg, master_clock_ms, ...)` — `ecg`:
- `(12, 5000)` 또는 `(2, 5000)`(= `[II, V2]`)를 받음. **추론은 항상 단일리드 II만** 사용
  (II→slot1, 나머지 0). V2는 로깅(raw 보존)·Phase-3 2리드 재검증 전용 — 2리드 동시추론 임계는 미검증.

---

## 2. 명세상 출력 계약 (spec — `records/00` §5-1)

> ⚠️ 이것은 **설계 명세**다. 실제 구현(§3)과 다르다.

```python
P1_output = {
    "cardiac_probs":   List[float],  # 길이 5, softmax 합=1.0
                                     # [NSR, AF, Ischemia, Conduction, Ectopic]
    "emergency_score": float,        # 0~1, sigmoid (AF+허혈 합산 확률)
    "embedding":       List[float],  # 길이 768, L2 정규화 없음 (raw mean-pool)
    "physio": {
        "hr_bpm": float,             # 60 / mean(R-R intervals@500Hz)
        "rhythm_regularity": float,  # 0~1; 명세 공식: 1 - clip(sdNN/200ms, 0, 1)
        # R-peak 미검출 시 둘 다 null
    },
    "model_version": str,            # "lora_multitask_snr_a07_e18"
    "inference_ms":  float,          # 단일 레코드 추론 시간
}
```

직렬화(명세 §5-2): 실시간 = JSON, 배치 = NumPy `.npz`(cardiac_probs Nx5, emergency_score N,
embedding Nx768, hr_bpm/rhythm_regularity N, record_ids, model_version).

---

## 3. 실제 구현 출력 (as-built — 코드 검증)

### 3.1 `P1CardiacChannel.infer()` 반환

```python
{
    "emergency_score": float | np.ndarray,   # sigmoid(BinaryHead(z))
    "cardiac_probs":   np.ndarray,           # softmax(MulticlassHead(z)), shape (5,) 또는 (N,5)
    "benign_flag":     bool | np.ndarray,    # argmax(cardiac_probs) ∈ {3,4}  → 비정상이나 양성
}
```
- **포함되지 않음**: `embedding`, `physio`, `model_version`, `inference_ms`.
- `embedding`은 내부에서 `emb = backbone(...)["x"].mean(1)` (768)로 **계산되나 반환되지 않고 폐기**된다.
- `benign_flag`는 **명세에 없는 신규 필드** (우세 유형이 전도장애·이소성이면 True).

### 3.2 `CardiacLoggingAdapter.to_record()` 반환 (§2 로깅 스키마)

```python
{
    "master_clock_ms": int,
    "ecg_lead_on":     bool,
    "emergency_score": float,
    "cardiac_p_nsr"/"cardiac_p_af"/"cardiac_p_isch"/"cardiac_p_cond"/"cardiac_p_ecto": float,
    "benign_flag":     bool,
    "hr_bpm":          float,   # estimate_physio() — R-peak 기반
    "rhythm_regularity": float, # 1 - clip(cv*3, 0, 1), cv = sdNN/mean  (※ 명세 공식과 다름, §4)
    "model_version":   "lora_multitask_snr_a07",   # ※ epoch 접미사 없음 (§4)
    # include_raw=True 시: "ecg_lead_II", "ecg_lead_V2" (float32 원신호)
}
```
- physio·benign_flag·model_version은 채우지만 **`embedding`은 여기서도 노출하지 않는다**.
- `estimate_physio`: `find_peaks(height=max(0.3·max,0.1), distance=0.3s)` → ≥2 peak이면
  `hr=60/mean(rr)`, `rhythm=clip(1-cv·3,0,1)`; 미검출 시 `(nan, nan)`.

---

## 4. 명세 ↔ 구현 드리프트 (P2 연결 전 확인 필수)

| 항목 | 명세(spec) | 구현(as-built) | 영향 |
|---|---|---|---|
| **`embedding[768]`** | 출력에 포함 (P2 융합 입력·핵심) | **어느 진입점에서도 미노출** (내부 계산 후 폐기) | 🔴 **P2 융합 입력 계약 미충족** — 노출 추가 필요 |
| `benign_flag` | 없음 | `infer`·`to_record`에 존재 | 신규 필드 — P2 소비 규약에 미반영 |
| `physio` 위치 | `infer` 출력의 중첩 dict | `infer`엔 없음. `to_record`가 flat 키로 제공 | 진입점별 상이 |
| `rhythm_regularity` 공식 | `1 - clip(sdNN/200ms)` | `1 - clip(cv·3)`, cv=sdNN/mean | 값 정의 불일치 |
| `model_version` | `"lora_multitask_snr_a07_e18"` | `"lora_multitask_snr_a07"` (접미사 없음) | 식별자 불일치 (명세 내부도 §5-1=`_e18` vs §5-2 예시=`_e22`로 비일관) |
| `inference_ms` | 포함 | 미구현 | 계측 미제공 |
| `physio_features.py`/`infer_p1.py` | 명세가 참조(미구현 명시) | 미존재 — physio는 adapter가 대체 구현 | 명세상 참조 파일과 실제 위치 상이 |

> 가장 중요한 것은 **`embedding` 미노출**이다. P2(멀티모달 융합)는 명세상 768-dim 임베딩을
> 융합 입력으로 받기로 되어 있으나, 현재 인터페이스는 이를 돌려주지 않는다.
> 통합 시 `infer()`/`to_record()`에 `embedding`을 추가 반환하도록 확장이 필요하다
> (백본 forward에서 이미 `emb`가 계산되므로 노출만 하면 됨 — 추가 연산 없음).

---

## 5. 출력 의미·인덱스 (검증됨)

- `cardiac_probs` 인덱스: `[0:NSR, 1:AF, 2:Ischemia(STD/STE), 3:Conduction(I-AVB/LBBB/RBBB), 4:Ectopic(PAC/PVC)]`.
- `emergency_score` = `sigmoid(BinaryHead(z))`. 학습 시 응급 라벨 = AF + 급성허혈(`EMERGENCY_CLASSES=(1,2)`).
- `benign_flag` = `argmax(cardiac_probs) ∈ {3,4}` (전도장애·이소성 우세 → 비정상이나 양성, 결정적).
- 임계 `tau_c`(응급 경보 컷)는 코드에 고정값이 없으며 **실데이터 튜닝 대상**
  (`p1_cardiac_channel.py` docstring: "cardiac 경보 if emergency_score ≥ tau_c").

---

## 6. P2 소비 규약 (명세 §5-3)

| P1 필드 | P2 사용 방식 |
|---|---|
| `emergency_score` | ECG 응급 채널(헤드라인) |
| `cardiac_probs` | 심장 원인 분류 → XAI 설명 |
| `embedding` | 융합 입력(768-dim 잠재 벡터) ※ 현재 미노출 — §4 |
| `physio.hr_bpm` | 자이로 활동량과 교차맥락. **null 허용**(다른 HR 소스로 대체) |

- 모달리티 가중은 P2 내부 신뢰도(conf) 게이팅이 담당(P1은 점수·임베딩을 그대로 전달).
- `physio`는 R-peak 미검출 시 `null`/`nan` 반환 — **P2에서 null 처리 필수**.

---

## 7. 사용 예 (코드 검증된 호출 형태)

```python
from p1_cardiac_channel import P1CardiacChannel
ch = P1CardiacChannel(device="cuda")          # 체크포인트 경로는 env P1_CKPT_FM / P1_CKPT_P1 로 재지정 가능
out = ch.infer(signal_12x5000)                # (12,5000) 또는 (N,12,5000)
# out = {"emergency_score", "cardiac_probs"(5), "benign_flag"}

from p1_cardiac_logging_adapter import CardiacLoggingAdapter
ad = CardiacLoggingAdapter()
rec = ad.to_record(ecg_2lead, master_clock_ms=t, include_raw=False)   # §2 로깅 레코드 dict
```

체크포인트 기본 경로(`p1_cardiac_channel.py`):
- 백본: `checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt` (env `P1_CKPT_FM`)
- P1 헤드+LoRA: `outputs/lora_multitask_snr_a07/lora_multitask_snr_best.pt` (env `P1_CKPT_P1`)
- 의존: `fairseq_signals.utils.checkpoint_utils.load_model_and_task` (백본 로드).
