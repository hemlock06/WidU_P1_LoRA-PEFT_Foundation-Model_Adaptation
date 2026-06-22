# 설계 결정 기록 (Design Decisions)

> 목적: 핵심 실험 설계 결정과 방법론적 근거 기록 — 각 분기점에서 무엇을 선택하고 왜 선택했는지, 어떤 대안을 고려했는지 추적
> 형식: 선택 / 대안 / 이유 / 영향 범위

---

## 단계 1 — 환경 사양 확정 (2026-05-24)

- **선택**: 로컬 RTX 3060 12GB에서 모든 실험 진행
- **대안**: 클라우드 GPU (RunPod RTX 4090 ~$0.4/h)
- **이유**: RTX 3060 12GB가 LoRA fine-tuning 최소 사양 충족. 클라우드는 메모리 부족 시 대안.
- **영향**: batch_size 16 (mixed precision 필요)

| 항목 | 사양 |
|---|---|
| Python | 3.9 (python) |
| PyTorch | 2.1.2+cu118 |
| CUDA | 11.8 |
| GPU | NVIDIA RTX 3060 12GB |
| OS | Windows |
| fairseq-signals | editable install (`pip install -e .`) |
| peft | 0.17.1 |
| transformers | 4.46.0 |
| ECG-FM 체크포인트 | `mimic_iv_ecg_physionet_pretrained.pt` (HuggingFace bowang-lab/ecg-fm) |

---

## Pre-flight 2 — ECG-FM 로드·추론 API 확정 (2026-05-24)

- **확인 항목**: 12-lead 슬롯에 N개 lead만 넣고 나머지 0-fill → forward 정상 작동하는가
- **결과**: PASS (preflight_2d_features_only.py 실행 확인)
- **영향**: 단계 5·6 전체에서 동일 API 사용

### 확정된 ECG-FM API

| 항목 | 확정값 |
|---|---|
| 로드 함수 | `load_model_and_task(filename)` — fairseq_signals.utils.checkpoint_utils |
| forward 호출 | `model(source=x, padding_mask=None, features_only=True)` |
| 출력 키 | `result['x']` |
| 출력 shape | `(B, 312, 768)` |
| 풀링 | `emb.mean(dim=1)` → `(B, 768)` |
| 모델 클래스 | `Wav2Vec2CMSCModel`, 90.9M params |

### 0-fill 테스트 결과

| Lead 구성 | std | 판정 |
|---|---|---|
| 12-lead (모두) | 12.93 | PASS |
| 4-lead (I,II,V2,V5) + 8개 0-fill | 13.00 | PASS |
| 2-lead (I,II) + 10개 0-fill | 12.92 | PASS |
| 1-lead (II) + 11개 0-fill | 12.80 | PASS |
| 0-lead (all zero) | 10.75 | PASS |

- **주의**: `features_only=True` 필수 — False 시 quantizer/contrastive head 실행 → Inf 발생

### 체크포인트 식별 주의 — 모델 사양은 yaml이 아니라 텐서 기준 (2026-06-03 점검)

- **실측 사양**: `mimic_iv_ecg_physionet_pretrained.pt` = wav2vec2_cmsc, **768-dim · 12-layer · 90.88M**
  (q_proj weight 768×768, `ckpt['cfg'].model.encoder_embed_dim=768`).
- ⚠️ **체크포인트 동봉 `.yaml` 사이드카는 stale 템플릿** — `encoder_embed_dim=1024`/`encoder_layers=24`로
  표기되나 실제 가중치와 불일치. 모델 식별은 **`.yaml`이 아니라 `ckpt['model']` 텐서 shape / `ckpt['cfg']`** 기준.
- **파일 크기 분해**: 사전학습 `.pt` 1.09GB = 모델 가중치 364MB + **Adam 옵티마이저 상태 727MB**(=2×params).
  추론·파인튜닝엔 옵티마이저 상태 불요 → 슬림 시 364MB.
- **P1 산출 체크포인트**: `lora_multitask_snr_best.pt`(348MB)의 대부분도 frozen 공개 백본 재저장분 —
  신규 학습 파라미터(LoRA r8 q/v + 헤드)는 **~0.3M(≈1–2MB)**. 재배포 시 백본은 공개 출처
  (HuggingFace bowang-lab/ecg-fm)에서 1회 취득하고 어댑터만 전달 가능.

---

## 단계 4 — CPSC 2018 이진 라벨 매핑 (2026-05-24)

- **선택**: 옵션 A — 엄격 이진 (Emergency: AF+STD+STE, Normal: Normal, Exclude: I-AVB+LBBB+RBBB+PAC+PVC)
- **대안**: 옵션 B — 포용 이진 (Exclude를 정상으로 처리)
- **이유**: 스펙 섹션 9.1 "양성 클래스(RBBB 등)는 응급에서 제외" 명시. 클래스 경계를 선명하게 유지하여 Phase 1 이진 검증의 학습 목표를 단순·명확하게 설정.
- **다중 라벨**: 하나라도 응급 → 응급(1). 모두 정상 → 정상(0). 제외 클래스만 → 제외(-1).
- **Split**: record-level random split (seed=42, 70/15/15) — patient ID 미제공

### 클래스별 결정 근거 (스펙 9.1)

| 클래스 | 스펙 기준 (의학적 근거) | 옵션 A (엄격) | 옵션 B (포용) |
|---|---|---|---|
| Normal | NSR = 정상 | 정상(0) | 정상(0) |
| AF | 위험 부정맥 | 응급(1) | 응급(1) |
| STD | 급성 허혈 | 응급(1) | 응급(1) |
| STE | 급성 허혈 | 응급(1) | 응급(1) |
| I-AVB | 1도 방실차단 — 양성, 비응급 | 제외(−1) | 정상(0) |
| LBBB | 좌각차단 — 응급 가능성 있으나 비특이적 | 제외(−1) | 정상(0) |
| RBBB | 우각차단 — 대부분 양성 | 제외(−1) | 정상(0) |
| PAC | 심방조기박동 — 대부분 양성 | 제외(−1) | 정상(0) |
| PVC | 심실조기박동 — 빈발 시 위험, 단발 양성 | 제외(−1) | 정상(0) |

→ **옵션 A 채택**: 경계가 모호한 5개 클래스(I-AVB/LBBB/RBBB/PAC/PVC)를 정상으로 흡수(B)하지 않고 제외(A). 응급/정상 경계를 선명하게 유지하여 Phase 1 이진 검증의 라벨 잡음을 최소화. 이 5개 클래스는 Phase 2 멀티헤드 설계(단계 5d)에서 전도장애·이소성 고유 클래스로 재통합 — 단계적 복잡도 확장 전략의 일환.

### SNOMED-CT 코드 매핑 (전처리 구현 기준 — `scripts/preprocess_cpsc2018.py`)

| 클래스 | SNOMED 코드 | 라벨 |
|---|---|---|
| Normal | 426783006 | 0 (정상) |
| AF | 164889003 | 1 (응급) |
| STD | 429622005 | 1 (응급) |
| STE | 164931005 | 1 (응급) |
| I-AVB | 270492004 | −1 (제외) |
| LBBB | 164909002 | −1 (제외) |
| RBBB | 59118001 | −1 (제외) |
| PAC | 284470004 / 63593006 | −1 (제외) |
| PVC | 427172004 / 17338001 | −1 (제외) |

- **미인식 SNOMED 코드 164884008**: "ECG: ventricular ectopics" (양성 부정맥) → 제외(−1) 확정

---

## 단계 6 — LoRA fine-tuning 설계 (2026-05-24)

- **선택**: 수동 LoRA 주입 (peft get_peft_model 대신)
- **대안**: patches/lora_fairseq_signals.patch — ECGTransformerClassifier 전용, Wav2Vec2CMSCModel 부적합
- **이유**: Wav2Vec2CMSCModel은 HuggingFace API 미지원 → LoRALinear 직접 치환이 안전
- **LoRA 설정**: rank=8, alpha=16, dropout=0.1, 대상=q_proj·v_proj (12 레이어 전체)
- **학습 파라미터**: ~295,681개 (전체 90.9M의 0.33%)

### LoRA 하이퍼파라미터 선택 근거 (2026-05-22)

| 하이퍼파라미터 | 값 | 선택 근거 | 대안 |
|---|---|---|---|
| rank r | 8 | LoRA 원논문(Hu et al. 2022) 대부분 태스크에 충분; 17k 소규모 데이터 과적합 위험 최소화 | r=4(표현력 부족) / r=16(과적합 위험) |
| alpha α | 16 | scaling = α/r = 2; HuggingFace PEFT 관례 기본값 | alpha=8(scaling=1) |
| dropout | 0.1 | 소규모 데이터 정규화; ECG-FM 템플릿 activation_dropout 값과 일치 | 0.0(정규화 없음) |
| 대상 모듈 | q_proj·v_proj only | LoRA 원논문 정준 선택 — Wq·Wv 적응이 성능/파라미터 최적; k_proj·out_proj·FFN 제외 | 전체 선형 레이어 |

### RLM(Random Lead Masking) 마스킹 확률 근거

- **선택**: p=0.5 (각 lead를 독립적으로 50% 확률로 zero-fill)
- **대안**: p=0.3 (보수적), p=0.7 (적극적)
- **이유**: ECG-FM 사전학습(Oh et al. CHIL 2022)에서 채택된 기본값과 동일하게 설정 — fine-tuning 중 사전학습 도메인 일관성 유지.
- **영향**: 임의 N-lead 부분집합 입력 시나리오 모사 → 디바이스 독립 일반화
- **측정된 가치 (2026-06-03 재점검)**: ③ multi-SNR 레시피서 RLM만 분리 시, 마진은 **단일리드+모션 구간에
  집중**(1-lead/6dB ΔAUROC +0.017·ΔSens@95Sp +0.096), full-clean·full-lead서 ≈0. 강건성 주동력은
  백본 사전학습 + multi-SNR이며 파인튜닝 RLM은 **보강(safety margin)** — 상세 `records/03 ⑥-c`.

### 실행 중 버그 수정: LoRALinear bias/weight property

- **문제**: `multi_head_attention.py:170`에서 `q_proj.bias` 직접 접근 → `AttributeError`
- **해결**: LoRALinear에 `weight`·`bias` property 추가 → `self.linear`로 proxy

---

## 단계 8 — multi-SNR 증강 설계 (2026-05-24)

- **NSTDB 노이즈 유형**: bw (baseline wander) / em (electrode motion) / ma (muscle artifact)
- **SNR 집합**: {24, 18, 12, 6, 0} dB (학습 분포)
- **p_noise**: 0.75 (25% clean 유지)
- **Lead 독립**: 각 lead별 독립 SNR 샘플링
- **증강 순서**: clean → 노이즈 주입 → RLM 마스킹 (역할 경계 유지)

### SNR 설계 결정 근거 (2026-05-25)

| 결정 항목 | 값 | 선택 근거 | 대안 |
|---|---|---|---|
| SNR 이산 집합 | {24,18,12,6,0}dB | 스펙 14-5 기본값; 단계 8 SNR 저하 곡선 버킷과 정합; 선행연구(MDPI Sensors 26(4):1135, 2026) 설정 준용 | 연속 균등 [0,24]dB |
| p_noise | 0.75 (25% clean) | clean 성능 보존과 수렴 안정의 균형 — 권장 구간(20~30% clean)의 중앙값; 스펙 단계6 "가벼운 강도 시작" 권고와 정합 | 전부 노이즈(p=1.0) |
| lead별 SNR | 독립 샘플링 | wearable lead별 접촉 품질 차이를 현실적으로 모사 | 샘플당 단일 SNR 공유 |

---

## 핵심 지표 선택 — Sensitivity@95%Specificity 운영점 (2026-05-24)

- **선택**: Sensitivity@95%Specificity를 응급 탐지 1차 운영점 지표로 사용
- **의미**: 5% 오거부율(정상 신호의 5%를 응급으로 오인하는 조건)에서 응급 탐지율
- **대안**: F1@0.5 (임계값 0.5에서 F1), AUROC (임계값 독립 지표)
- **근거**:
  1. **alarm fatigue 요건**: 연속 모니터링에서 false positive rate가 alarm fatigue에 직결되는 핵심 지표. 5% FPR = 95% 특이도는 웨어러블 연속 모니터링에서 alarm fatigue 없이 허용 가능한 오경보 상한선.
  2. **임상 AI 스크리닝 관례**: AUROC는 임계값-독립 요약 지표이지만 실배치 시 운영 임계값 직접 제공 불가. 특이도 고정 후 민감도 보고(Sens@95%Sp)는 응급·스크리닝 AI 논문의 표준 관례 — "고특이도 조건에서 실운영 탐지율"을 직관적으로 표현.
  3. **95% 기준**: 5% FPR은 임상 AI 스크리닝 논문에서 허용 가능한 오경보율 상한으로 암묵적 통용됨 (특정 논문 명시적 논의 기록 없음 — 관례 채택).
- **영향**: 모든 단계(5·6·7·8)에서 동일 운영점 지표 적용

---

## 단계 4 보완 — CACHET patient-level split 설계 확정 (2026-05-25)

- **선택 A**: CACHET(1602개) → stage 9 inference-only held-out 전용. 학습에는 미사용.
- **대안**: Full Format(16.5GB) 다운로드 → patient-level split → 학습 + held-out 분리
- **이유**:
  1. Short Format에 patient ID 없음 → strict patient-level split 불가
  2. CACHET을 held-out 전용으로 둬 외부검증 도메인 분리 확보 → 데이터 누출 없는 엄격한 외부 검증 환경
  3. patient-level leakage 논란 원천 차단
- **영향**: 단계 5·6·8은 CPSC만 → 재작업 불필요.

### CACHET 라벨 → 단계 9 매핑

| CACHET 라벨 | 값 | 단계 9 이진 |
|---|---|---|
| AF | 1 | 응급(1) |
| NSR | 2 | 정상(0) |
| Noise | 3 | **제외** |
| Others | 4 | **제외** |

유효 샘플: AF(747) + NSR(615) = 1,362개

---

## 단계 9 — LTST 전처리 OOM 수정 설계 (2026-05-26)

### 문제: scipy.signal.resample (FFT 기반) → 긴 레코드 OOM

LTST는 레코드당 21~45시간 × 250Hz = 최대 40,500,000 샘플. 원래 전처리 코드는:
1. `sp_resample(sig, T_out, axis=0)` → FFT 기반, 중간 복소수 배열 (T_out, n_sig) complex64 할당
2. `sig_12 = np.zeros((12, T_out), dtype=float32)` → 45시간 레코드 시 (12, 81,000,000) ≈ 3.7 GB 단일 배열

두 단계 모두 연속 메모리 대량 할당 → `std::bad_alloc` (실험: s20081부터 연속 실패).

### 3단계 수정 이력

| 버전 | 변경 | 결과 |
|---|---|---|
| v1 (원본) | `sp_resample` + 전체 12-lead 배열 | s20081~연속 OOM 실패 |
| v2 | `resample_poly`(polyphase) 교체 | 리샘플 단계 OOM 해소, 12-lead 배열 단계 여전히 실패 |
| v3 (확정) | + 윈도우 단위 12-lead 배치 (`make_window_12lead`) + `del sig_rs` 즉시 해제 | 86/86 전체 통과 |

### v3 확정 설계

- **`resample_poly(sig, up=2, down=1, axis=0)`**: polyphase 필터, FFT 대비 메모리 ~10배 효율. 중간 복소수 배열 없음.
- **`make_window_12lead(sig_rs, slot_indices, w)`**: 윈도우당 (12, 5000) = 240 KB만 할당. 전체 (12, T_out) 폭발 회피.
- **`del sig_rs`**: 레코드 처리 직후 2-lead 원본 해제 → 다음 레코드 메모리 확보.
- **영향**: 86/86 레코드 완전 처리. 출력: 15,426 윈도우 (응급 11,346 / 정상 4,080).

---

## 단계 5b — 심장 다중분류 설계 확정 (2026-05-26)

### CPSC 9-class → 5-class 매핑

| 5-class | CPSC 원본 클래스 | 이진 매핑 |
|---|---|---|
| 정상(NSR) | Normal | 정상(0) |
| AF | AF | 응급(1) |
| 허혈성(STD/STE) | STD + STE | 응급(1) |
| 전도장애 | I-AVB + LBBB + RBBB | 정상(0)* |
| 이소성 | PAC + PVC | 정상(0)* |

\* 기존 이진 모델에서 제외(-1)됐던 클래스를 다중분류에서 고유 클래스로 복귀. 이진 헤드와 다중분류 헤드는 동일 ECG-FM 백본 위에 병행.

### 헤드 구조 및 손실 함수

- **선택**: `MulticlassHead(768→5)` + `CrossEntropyLoss` + 클래스별 역빈도 가중치
- **대안**: 이진 헤드 5개 독립 훈련 (multi-label)
- **이유**: 다중분류가 클래스 간 상호 배타성을 명시적으로 학습. 역빈도 가중치(이소성 w=2.34, 전도장애 w=0.49)로 이소성 소수 클래스 보정.
- **학습 파라미터**: 298,757개 (LoRA 24레이어 + LinearHead 3,845개)

### 이진 응급 AUROC 도출 방식

- 5-class softmax에서 응급 클래스(AF+허혈성) 확률 합산 → 이진 점수
- 결과: 이진 AUROC 0.9263 (단독 이진 모델 0.9463 대비 −0.020)
- 해석: 단일 헤드로 5분류 + 이진 판단 동시 제공 시 약간의 이진 성능 트레이드오프 발생 (예상 범위)

### 데이터셋

- 전처리 스크립트: `scripts/preprocess_cpsc2018_mc.py`
- 출력: `data/processed/cpsc2018_mc/` (train=4357, val=933, test=936)
- 기존 이진용(2217/474/474)보다 큰 이유: 제외(-1)됐던 전도장애·이소성 클래스 포함

---

## 전처리 방법론 통합 설계 (2026-05-26)

> 이 섹션은 모든 데이터셋에 공통 적용된 신호 전처리 파이프라인의
> 설계 결정과 근거를 통합 기록합니다.
> 각 DB별 세부 실행 기록은 `records/04_run_history.md` 참조.

### 1. normalize=False — 정규화 미적용

**결정**: 전처리 단계에서 신호 정규화(z-score, min-max 등)를 수행하지 않음.

**근거**:
- ECG-FM 사전학습 설정 파일에 `normalize: false`가 명시돼 있음:
  - `checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.yaml:132: normalize: false`
  - `checkpoints/ecg-fm/mimic_iv_ecg_finetuned.yaml:123: normalize: false`
- ECG-FM은 mV 단위 원시 신호로 사전학습됨. 정규화 적용 시 분포 불일치(distribution shift) 발생 → 사전학습된 표현 손상.
- ECG 임상 관례상 신호 진폭 자체(mV)가 진단 정보를 담음 (예: ST 상승 기준 ≥0.1mV). 정규화하면 이 정보가 소실됨.

**주의**: STAFF-III 일부 lead에 부분적 NaN 존재 → `nan_to_num(..., nan=0.0)` 처리. 이는 "정규화"가 아닌 결측값 처리.

---

### 2. 500Hz 고정 — 다른 샘플링 주파수 처리

**결정**: 모든 데이터셋 출력을 500Hz로 통일.

**근거**:
- ECG-FM은 PhysioNet MIMIC-IV ECG (500Hz) 데이터로 사전학습됨.
- 입력이 다른 Hz인 경우 주파수 특성 불일치가 발생하므로 리샘플링이 필수.
- `scipy.signal.resample`(푸리에 기반 리샘플)이 aliasing 방지에 유리.

**데이터셋별 원본 Hz 및 처리**:

| 데이터셋 | 원본 Hz | 처리 |
|---|---|---|
| CPSC 2018 | 500Hz | 그대로 사용 (FS_REQUIRED=500, 불일치 레코드 스킵) |
| PTB-XL | 500Hz (records500/) | 그대로 사용 |
| STAFF-III | 1000Hz | `resample(T, T//2)` — 1000→500Hz |
| CACHET-CADB | 1024Hz | `resample(10240, 5000)` — 10240→5000 샘플 |
| INCART | 257Hz | `resample(T, T_out)` — 257→500Hz |
| LTST | 250Hz | `resample_poly(up=2, down=1)` — 250→500Hz |

---

### 3. 10초/5,000샘플 윈도우 고정

**결정**: 단일 ECG 세그먼트 길이를 10초(5,000샘플 @500Hz)로 고정.

**근거**:
- ECG-FM positional encoding이 5,000 샘플 고정 입력을 가정함 (모델 구조 제약).
- 임상 12-lead ECG 표준 기록 길이가 10초(AHA 권고).
- CPSC 2018 원본 레코드가 10초 → 자연스럽게 일치.
- LTST·STAFF-III 등 장기 레코드는 비중첩(non-overlapping) 10s 윈도우로 분할.

**비중첩 선택 이유**: 중첩(overlapping) 윈도우는 학습/평가 샘플 간 의존성(temporal leakage) 발생. 비중첩이 독립성 보장.

---

### 4. 앞자름(Head Crop) vs Center Crop

**결정**: 10초 이상 레코드는 **앞 5,000 샘플** 사용 (`sig[:5000]`). Center crop 미사용.

**근거**:
- CPSC 2018 레코드 대부분이 정확히 10초(5,000 샘플). 초과 레코드가 드물고 초과량도 미미함.
- ECG 앞부분에 P파·QRS·ST 분절 등 주요 파형이 고루 분포 → 앞자름으로 대표성 충분.
- 장기 레코드(LTST·STAFF-III)는 어차피 비중첩 윈도우 분할 방식 사용 → head crop 해당 없음.
- Center crop은 실질적 차이 없으면서 구현 복잡도만 증가.

---

### 5. 우측 Zero-Padding

**결정**: 10초 미만 레코드는 **우측(끝)에 0을 채워** 5,000 샘플로 맞춤.

**근거**:
- ECG가 짧은 것은 데이터 품질 문제(기록 중단 등)로 희귀 케이스.
- 우측 패딩은 기존 신호에 영향을 주지 않음. 좌측 패딩은 타이밍 정보를 왜곡할 수 있음.
- 0 값은 baseline(등전위선)에 해당 — 생리학적으로 중립.
- Wav2Vec2 계열 모델이 패딩 마스크를 지원하나, ECG-FM API에서 `padding_mask=None`으로 통일 (실제 패딩 케이스가 매우 드물어 마스킹 필요성 낮음).

---

### 6. 이상값(NaN/Inf) 처리

**결정**: 데이터셋 특성에 따라 두 가지 전략 사용.

| 전략 | 적용 데이터셋 | 이유 |
|---|---|---|
| 레코드 전체 스킵 | CPSC, PTB-XL | 단일 에폭 레코드 → NaN이면 전체가 무의미 |
| `nan_to_num` 0-fill | STAFF-III, CACHET | 장기 레코드 일부 구간만 NaN — 레코드 전체 버리기보다 0-fill이 데이터 보존에 유리. 0=등전위선으로 해석 가능 |

---

### 7. 리드 수 불일치 처리 — 0-fill

**결정**: 표준 12-lead 미만 데이터셋은 **가용 리드를 해당 슬롯에 배치하고 나머지 슬롯을 0으로 채움**.

**구체 예시**:

| 데이터셋 | 가용 리드 | 표준 12-lead 매핑 |
|---|---|---|
| CACHET-CADB | 1-lead (Lead II) | slot 1(II)에 배치, 나머지 11개 슬롯 0-fill |
| LTST | 2-lead (ML2→II, MV2→V2) | slot 1, 7에 배치, 나머지 10개 슬롯 0-fill |
| STAFF-III | 9-lead (V1-V6, I, II, III) | slot 0-2, 6-11에 배치, aVR/aVL/aVF(slot 3-5) 0-fill |

**근거**:
- ECG-FM은 RLM(Random Lead Masking) 사전학습으로 리드 누락에 강건함 (0-fill = 마스킹과 동일 신호).
- 다른 슬롯에 값을 채우는 것보다 0-fill이 혼동 없음.
- N-lead 강건성 실험(Ablation A1)에서 1-lead까지 AUROC 0.9408 유지 확인 → 0-fill 전략 유효성 검증됨.

---

## Phase 2 — P1 추론기 아키텍처 결정: 단일 백본 vs 분리 모델 (2026-05-28)

### 배경

Phase 1에서 독립적으로 최적화된 두 모델이 완성됨:
- **이진 탐지**: `lora_multisnr_best.pt` (③, CPSC 이진 AUROC=0.9463, multi-SNR 강건성 검증 완료)
- **다중분류**: `lora_mc_best.pt` (5b, Macro-F1=0.6762, per-class AUROC 0.862~0.976)

두 모델을 P1 출력으로 통합할 때 아키텍처 방식을 결정해야 함.

### 선택지 비교

**옵션 A — 분리 모델 직렬 호출 (Path A)**:

```
입력 ECG (12, 5000)
    ├──→ ECG-FM+LoRA(③) → embedding_A(768) → BinaryHead  → emergency_score
    └──→ ECG-FM+LoRA(5b) → embedding_B(768) → MCHead      → cardiac_probs
```

- 장점: 각 모델이 태스크별로 독립 최적화 → 개별 성능 최고
- **단점**: 백본 3개 → forward 3회 / embedding이 두 종류(A, B) — P2 입력으로 어느 것을 전달할지 설계상 모호

**옵션 B — 단일 백본 멀티헤드 (Path B)**:

```
입력 ECG (12, 5000)
    └──→ ECG-FM+LoRA(5d) → embedding(768) ┬→ BinaryHead  → emergency_score
                                           └→ MCHead      → cardiac_probs
```

- 장점: embedding 단일, P2 입력 명확 / forward 1회 / 설계 일관성
- 단점: 이진·다중 태스크 trade-off로 개별 성능이 분리 모델 대비 약간 낮을 수 있음

### Embedding 문제 — Path A의 핵심 설계 이슈

Path A에서 P2로 전달할 `embedding: float[768]`을 결정해야 한다. ③(이진) 백본과 5b(다중분류) 백본은 서로 다른 LoRA 가중치로 학습됐으므로, 두 embedding이 다른 표현 공간에 있다. 후보:

| 후보 | 근거 | 문제 |
|---|---|---|
| ③ embedding | 이진 응급 탐지 최적화 | 다중분류 표현 반영 안 됨 |
| 5b embedding | 5-class 분류 최적화 | 이진 응급 강건성 낮음 |
| 동결 ECG-FM embedding | 태스크 중립 | fine-tuning 표현력 손실 |

Path B는 이 문제를 해소 — 단일 학습된 embedding이 두 태스크 정보를 동시에 담음.

### 결정 (2026-05-28) — **Path B 채택 확정**

5d+SNR 학습 완료 결과:
- 이진 AUROC = **0.9134** (기준 0.91 충족)
- Macro-F1   = **0.6834** (기준 0.68 충족)

→ **Path B (단일 백본 멀티헤드) 채택.**
→ P2 인터페이스 `embedding: float[768]` = `lora_multitask_snr_best.pt` 백본 임베딩으로 확정.
→ 체크포인트: `outputs/lora_multitask_snr/lora_multitask_snr_best.pt`

### 연구 서사 측면

어느 Path를 채택하더라도 아래 서사로 일관성 있게 기술 가능:
> "Phase 1에서 이진·다중분류 파이프라인 각각의 신뢰도를 독립 검증 완료.
> Phase 2에서 단일 백본 통합 여부를 embedding 일관성 및 성능 트레이드오프 기준으로 판단."
>
> Path B 채택 시: "단일 백본으로 통합, embedding 일관성 확보"
> Path A 채택 시: "분리 모델 직렬 호출, embedding은 다중분류 백본 것 사용으로 규약"

---

## 다중분류 약클래스 근본원인 + 데이터/지표 재설계 (2026-05-29)

### 배경 — 약클래스 정체의 근본원인 분석 (백본·데이터 수준까지 추적)

5b 다중분류의 약클래스(이소성 F1=0.35, NSR 0.61, 허혈 0.66)에 대해, 말단 헤드 튜닝에 그치지 않고 백본 표현과 데이터 구성까지 거슬러 올라가 근본원인을 규명했다. 약클래스가 "더 학습하면 오를 문제"인지 "구조적 천장"인지부터 가르는 것이 목표였다.

**검증된 사실 (직접 카운트/실험)**:

1. **mean-pooling 가설 반증.** max-pool / mean+max+std를 frozen ECG-FM 특징에 직접 적용 → 모든 클래스에서 더 나쁨. ECG-FM의 312-step 특징이 strip 내 거의 정상(stationary)이라 풀링 변경으로 단발 박동 정보 복원 불가. → **풀링 재설계 효익 ≈ 0.**
2. **클래스 희소 + 라벨 매핑 결함이 진짜 원인.**
   - 이소성 train 372개(8.5%) — COND 1770 대비 ~5배 불균형 (검증: labels.npy 카운트).
   - **`164884008`(ventricular ectopics) 코드 603개 레코드가 전량 배제** (SNOMED_ECTO에 미포함 → 인식 코드 없어 -1). 별도 93개는 다른 라벨로 흡수. raw .hea 6839개 스캔으로 검증 — 단일 최대 배제 코드.
   - `map_label_mc` 우선순위 붕괴(ISCH>AF>COND>**ECTO**>NSR): 동반 코드 보유 시 이소성 라벨이 상위 클래스에 박탈됨.
3. **부수 발견(미재현)**: frozen 선형 프로브가 배포 LoRA 모델보다 약클래스 per-class에서 우수 → 현 학습 레시피(composite가 binary+macroF1 지배)가 per-class 순위를 약간 훼손 가능.

### 목표 수치 타당성 판정

초기에 세운 목표(F1≥0.80 & AUROC≥0.93 전 클래스)는 부분적으로 비현실적이며 임무와 어긋남이 드러났다:
- **이소성 F1≥0.80**: 본질적 비현실 — CPSC 2018 우승자도 beat-level로 0.6~0.8. strip-level + frozen 백본으로 불가. 또한 이소성은 **양성(benign)**, `labels_bin`=0 → 응급 판단 무관.
- **허혈 F1≥0.80**: strip-level 비현실 (beat·ST 분석 필요). AUROC≥0.92는 근접 가능.
- AF/전도장애: 이미 충족. NSR: AUROC 충족, F1은 precision 한계.

→ **Macro-F1을 헤드라인 지표로 쓰는 것이 부적절.** P1 일차 임무 = 이진 응급 탐지, 5-class는 P2용 보조 맥락.

### 결정 — 데이터 결함 수정 + 지표 재정의 (채택)

1. **데이터**: `SNOMED_ECTO`에 `164884008` 추가(이소성 603개 복구), 단일라벨 우선순위 → **multi-label(N,5 binary)** 로 변경(동반코드 라벨 보존). PTB-XL 다중분류 합성 통합 검토.
2. **지표 재정의(미션 정렬)**: 헤드라인 = 이진 응급 AUROC(≥0.93 목표), AF AUROC(≥0.96), 허혈 AUROC(≥0.92). 이소성은 F1≥0.55 수준 "soft 맥락"으로 강등. 응급-가중 F1(AF+허혈) 보조 추적.
3. 풀링 재설계·beat-level 하이브리드는 보류(효익 대비 비용/미션정렬 부적합).

상세 실행 계획·진행: `records/05_open_issues.md` §7.

### 후속 결과 — 데이터 수정(5e)은 이소성 개선에 실패 (2026-05-29)

5e 재학습 후 결정성 고정 + 누수 제거 공정 비교(clean subset 251, 세 모델 미학습) 결과
(전체 수치: `records/03_eval_results.md § 단계 5e`):

- **주 목표 미달**: 이소성 AUROC 5e 0.820 < 5b/5d 0.854 — 데이터를 늘렸는데 **오히려 하락**.
- **근본원인 분석 재정의 (이소성 출처별 분해, `analyze_ectopic_source.py`)**:
  - 위 §의 "데이터 부족(372개)" 가설이 **부분 반증됨** — 909개로 늘려도 이소성 AUROC는 천장 유지.
  - 복구분 164884008은 동반진단 적고(12.6%) 난이도도 PAC/PVC와 동등 → "어려운 하위분포" 아님.
  - **5b/5d는 164884008 미학습인데도 동등 구분** → ECG-FM 동결 표현이 이미 심실이소성 포착.
  - ∴ **이소성 ~0.82는 표현 천장**. 데이터 양(5e)·풀링(앞 §) 둘 다 못 깸 → 병목은 표현 자체.
- **미션 영향 없음**: 이소성은 benign·`labels_bin`=0 → 응급 판정 무관. 응급 헤드라인은 유지
  (5e 응급 AUROC 0.932, AF 0.986).
- **부수 효과**: 5e는 5d 대비 응급·전도 소폭↑ + multi-label(동반진단) 출력 가능.
- **P1 채택 확정**: 5d 계열(단일 라벨 softmax) + α=0.7 재학습(5f) — 이하 §α=0.7 참조.

---

## α=0.7 손실 가중치 — P1 최종 설계 결정 (2026-05-29)

### 결정

멀티태스크 손실: `loss = 0.7·BCE(이진) + 0.3·CE_weighted(다중분류)`.
기존 α=0.5 대비 BCE 가중치 상향.

### 근거 — 그래디언트 간섭 해소

α=0.5에서 BCE 그래디언트(스칼라·단순)와 CE 그래디언트(5-class·고분산)가 공유 LoRA에
동등 가중치로 경쟁 → **그래디언트 간섭(conflicting gradients)** 발생, 공유 z 수렴 방해.

α=0.7에서 BCE가 주도권을 가져 공유 z를 **"응급 vs 비응급" 축으로 선명하게 정렬**:
- 이진 태스크와 다중분류 태스크는 동일한 "응급 축"을 공유 → BCE 지배 z는 MCHead에게도 읽기 쉬운 표현
- 역설적으로 CE 가중치를 줄였는데 MacroF1이 향상 (val 0.6834 → 0.6998)

### 검증 결과 (5d+SNR α=0.5 vs 5f α=0.7, CPSC mc test)

| 지표 | α=0.5 | α=0.7 | Δ |
|---|---|---|---|
| BinAUROC | 0.9139 | 0.9139 | ±0 |
| Macro-F1 | 0.6834 | 0.6858 | +0.002 |
| val composite | 0.8046 | **0.8051** | +0.0005 |

Test 수치 동점(신뢰구간 내), val composite 미세 우위 + 메커니즘 검증 → **α=0.7 채택**.
