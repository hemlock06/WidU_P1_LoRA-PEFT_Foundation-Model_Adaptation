# ARCHITECTURE — ECG-FM 응급 심장 이상 탐지 (P1)

> 개발자·통합자 인수인계용 아키텍처 문서. **레포의 실제 코드·체크포인트·records 기록을
> 직접 읽어 검증한 사실만** 기술한다. 애매하거나 미검증인 부분은 명시적으로 표시했다.
> 작성 시점 기준 브랜치: `handoff-prep` (소스 로직 변경 없음, 문서 추가만).
>
> 1차 출처: `scripts/train_lora_multitask.py`, `scripts/p1_cardiac_channel.py`,
> `scripts/multisnr.py`, `records/ecgfm_backbone_spec.md`, `records/01_design_decisions.md`.

---

## 1. 전체 개요

정적 임상 12-lead로 사전학습된 ECG Foundation Model(ECG-FM)을 **frozen** 상태로 두고,
**LoRA(q_proj·v_proj)** 로만 적응하여 웨어러블 환경의 심혈관 응급(AF·급성허혈)을 탐지한다.
단일 백본 위에 **이진 응급 헤드 + 5-class 심장 분류 헤드**를 병행(멀티태스크)하고,
학습 시 **multi-SNR 모션 증강 + RLM 가변 lead 마스킹**으로 노이즈·lead 부족에 강건하게 만든다.

```
ECG 입력 (12-lead · 500Hz · 10s = 5000샘플 · raw mV · N-lead는 0-fill)
          │
          ▼
  ECG-FM (wav2vec2_cmsc, 90.9M, frozen)
   ├ CNN feature extractor (4블록, 256ch, k2/s2 → 16× 다운샘플 → 312 토큰)
   ├ post_extract_proj  Linear(256→768)
   ├ conv positional embedding (Conv1d k=128, groups=16)
   └ Transformer encoder × 12층 (dim=768, heads=12, FFN=3072, Post-LN)
      └ self_attn.q_proj / v_proj 에 LoRA 주입 (rank=8, α=16)
          │
          ▼
  mean-pool over time  (B,312,768) → (B,768)  =  embedding z
          │
          ├── BinaryHead     Linear(768→1) → sigmoid → emergency_score
          └── MulticlassHead Linear(768→5) → softmax → cardiac_probs[5]
```

**P1 확정 모델**: `outputs/lora_multitask_snr_a07/lora_multitask_snr_best.pt` (α=0.7, epoch 18).
검증 출처: `decisions.md`, `records/02_training_logs.md` ⑧, `records/03_eval_results.md` §5f.

---

## 2. 백본 — ECG-FM (frozen)

출처: `records/ecgfm_backbone_spec.md` (체크포인트 `cfg.model` + `named_modules()` 실측).

| 항목 | 실측값 | 비고 |
|---|---|---|
| `_name` | `wav2vec2_cmsc` | Wav2Vec2 + CMSC 사전학습 |
| 파라미터 | 90.9M | frozen (전부 `requires_grad_(False)`) |
| 입력 | 12-lead raw mV, `normalize=False` | 인스턴스 정규화 없음 |
| CNN feature extractor | Conv1d 4블록, out=256, k=2, s=2 (bias 없음) | 다운샘플 2⁴ = **16×** |
| 시간 토큰 수 | 5000샘플 → **312 토큰** | 5000/16 = 312.5 → 312 |
| feature projection | `post_extract_proj` Linear(256→768) | conv 출력 256 → 인코더 768 |
| positional | convolutional (Conv1d k=128, groups=16) | 사인파/absolute 아님 |
| encoder | 12층, dim=768, heads=12(head dim 64), FFN=3072 | |
| norm 방식 | **Post-LN** (`layer_norm_first=False`) | norm을 residual add 이후 적용 |
| 활성 | GELU | |

> ⚠️ **주의 (stale 메타데이터)**: 체크포인트 옆 `.yaml` 사이드카
> (`mimic_iv_ecg_physionet_pretrained.yaml`)는 차원을 **1024/24-layer로 오기**하고 있다.
> 실제 값은 ckpt `cfg`·텐서 기준 **768-dim / 12-layer**다. 도식·문서 작성 시 yaml을 신뢰하지 말 것.
> (출처: `records/ecgfm_backbone_spec.md` 머리말 ★)

### 체크포인트 2종
`checkpoints/ecg-fm/`에 두 개가 있으나 P1 코드가 백본으로 쓰는 것은 **pretrained 쪽**이다.

| 파일 | 용도 | P1 사용 |
|---|---|---|
| `mimic_iv_ecg_physionet_pretrained.pt` | 사전학습 백본 (CMSC) | ✅ (`p1_cardiac_channel.py` `_DEF_FM`, `train_*` `CKPT_FM`) |
| `mimic_iv_ecg_finetuned.pt` | 사전학습 측 별도 파인튜닝 가중치 | ❌ P1 스크립트에서 미참조 |

---

## 3. LoRA 적응층

출처: `scripts/train_lora_multitask.py` (`LoRALinear`, `inject_lora`), `scripts/p1_cardiac_channel.py` (`LoRALinear`, `_inject`).

| 항목 | 값 |
|---|---|
| rank `r` | 8 |
| α | 16 → scaling = α/r = **2.0** |
| dropout | 학습 0.1 / 추론 0.0 |
| 대상 모듈 | `self_attn.q_proj`, `self_attn.v_proj` **만** (k_proj·out_proj·FFN 미부착) |
| 학습 파라미터 | ~295K = 백본 90.9M의 **0.33%** (records 기록) |
| 초기화 | `lora_A` kaiming_uniform, `lora_B` zeros (초기 출력 = 백본 그대로) |
| forward | `original(x) + lora_B(lora_A(dropout(x))) · scaling` |

LoRA 부착 방식: 런타임에 `named_modules()`를 순회하며 대상 `nn.Linear`를
`LoRALinear`로 **monkeypatch 교체**한다(`inject_lora` / `_inject`). 학습 결과는
체크포인트의 `backbone_lora` state dict(키에 `lora_` 포함)로 저장된다.

> ⚠️ **인수인계 주의 — LoRA 주입 경로가 2가지**: 위 런타임 monkeypatch와 별개로,
> `patches/lora_fairseq_signals.patch`는 **fairseq-signals 라이브러리 내부**
> (`multi_head_attention.py`, `ecg_transformer_classifier.py`)에 LoRA를 주입하는 패치를 보존한다.
> 현재 학습·추론 스크립트(및 P1 확정 체크포인트)가 실제로 쓰는 경로는 **런타임 monkeypatch**로 보이며,
> 라이브러리 패치는 별도/이전 경로일 가능성이 있다. **어느 쪽이 정준(canonical)인지는 미검증** —
> 통합 전 확인 필요(→ `HANDOFF_ISSUES.md` P1).

---

## 4. 멀티태스크 헤드

출처: `scripts/train_lora_multitask.py` `BinaryHead`, `MulticlassHead`; `scripts/p1_cardiac_channel.py` `_Head`.

| 헤드 | 구조 | 출력 | 활성 |
|---|---|---|---|
| BinaryHead | Linear(768→1) | `emergency_score` (스칼라) | sigmoid |
| MulticlassHead | Linear(768→5) | `cardiac_probs[5]` | softmax |

- 공유 입력 = mean-pool 임베딩 `z` (768). **forward 1회 → embedding 1개 → 두 헤드 동시 출력**.
- 5-class 인덱스: `[0:NSR, 1:AF, 2:Ischemia(STD/STE), 3:Conduction(I-AVB/LBBB/RBBB), 4:Ectopic(PAC/PVC)]`.
- 응급 클래스 정의: `EMERGENCY_CLASSES = (1, 2)` (AF + 급성허혈).

> **Path B(단일 백본 멀티헤드) 채택 근거** (`records/01_design_decisions.md`):
> 분리 모델 직렬(Path A)은 임베딩이 2개가 되어 P2 인터페이스가 모호해진다.
> 단일 백본은 forward 1회·임베딩 단일성·설계 일관성을 확보한다. 대신 단독 이진 모델 대비
> 이진 AUROC가 트레이드오프로 하락한다(0.9463 → 0.9139, §결과 참조).

---

## 5. 학습 증강 파이프라인

출처: `scripts/multisnr.py`, `scripts/train_lora_multitask.py` (`random_lead_mask`, train 루프).

**파이프라인 순서 (스펙 8-5, train 루프 ①②)**:
```
clean 배치 → ① multi-SNR 노이즈 주입 → ② RLM 마스킹 → ECG-FM+LoRA → heads
```
노이즈를 먼저, 마스킹을 나중에 두어 각 lead가 {노이즈 신호} 또는 {0=부재}로 분리된다.

### 5.1 multi-SNR 모션 증강 (`MultiSNRNoise`)

| 항목 | 값 |
|---|---|
| 노이즈 소스 | NSTDB (MIT-BIH Noise Stress Test): `bw`(baseline wander)·`em`(electrode motion)·`ma`(muscle artifact) |
| 리샘플 | NSTDB 360Hz → 500Hz (`resample_poly`, up=25/down=18), 2채널 flat 1D pool |
| SNR 집합 | {24, 18, 12, 6, 0} dB, 균등 추출 |
| `p_noise` | 0.75 (샘플의 25%는 clean 유지) |
| per-lead | 각 lead 독립 SNR·구간 샘플링 |
| noise_weights | (bw, em, ma) = (0.25, 0.50, 0.25) — em 강조 |
| 주입 수식 | `α = sqrt(P_sig / (P_noise · 10^(SNR/10)))`, `x_noisy = x + α·n` (unit-invariant) |

**noise_mode 3종**:
- `single` (코드 기본값): lead마다 bw/em/ma 중 1종 선택.
- `mixed`: lead마다 bw·em·ma를 std 정규화 후 가중합(동시 중첩) 1회 주입.
- `mixed_temporal`: mixed + 종류별 시간 엔벨로프(bw 지속 / ma 빈번 / em burst). **배포 학습 권장 레시피**
  (`records/03_eval_results.md` ⑩; 단 코드 기본값은 재현성 안전을 위해 `single` 유지).

> 평가용 `inject_fixed(snr_db)`는 전 lead에 고정 SNR을 주입(SNR 저하 곡선 전용, 학습 미사용).

### 5.2 RLM (Random Lead Masking)

출처: `train_lora_multitask.py` `random_lead_mask(x, p=0.5)`.
- 각 lead를 독립적으로 `p=0.5` 확률로 zero-fill (`mask = (rand > p)`).
- 근거(`records/01`): ECG-FM 사전학습(Oh et al. CHIL 2022) 기본값과 동일 → 도메인 일관성, N-lead 일반화.
- 정직한 역할 평가: 주동력은 백본+multi-SNR이고, RLM은 단일리드+중간 모션에서 보강(safety margin) 역할
  (`records/03` ⑥-c).

### 5.3 손실

`loss = α · BCE(이진) + (1−α) · CE_weighted(다중)`

| 항목 | 값 |
|---|---|
| α (BCE 비중) | **0.7** (P1 확정). 코드 기본값은 0.5 |
| BCE | `BCEWithLogitsLoss(pos_weight = n_neg/n_pos)` |
| CE | `CrossEntropyLoss(weight = 역빈도 클래스 가중)` |
| optimizer | AdamW (lr 1e-4, weight_decay 1e-2), CosineAnnealingLR (eta_min = lr·0.1) |
| grad clip | max_norm 1.0 |
| best 선택 | `composite = (val bin_AUROC + val macro_F1) / 2` |
| 기타 | mmap 데이터셋(OOM 방지), self-healing resume(rng 상태 포함 결정적 재개) |

> α=0.7 근거(`records/01`, `records/03` §5f): α=0.5에서 BCE/CE 그래디언트가 공유 LoRA에서 경쟁.
> α=0.7로 BCE 우위 → z를 "응급 vs 비응급" 축으로 정렬 → val Macro-F1 향상. test는 사실상 동점(±0.002 비결정성 내).

---

## 6. 모듈 맵 (`scripts/`)

| 카테고리 | 파일 | 역할 |
|---|---|---|
| **추론/통합 진입점** | `p1_cardiac_channel.py` | 검증된 단일 추론 진입점(`P1CardiacChannel.infer`). 전처리·LoRA·헤드 캡슐화 |
| | `p1_cardiac_logging_adapter.py` | 추론+physio → 로깅 §2 레코드(`CardiacLoggingAdapter.to_record`) |
| **학습** | `train_lora_multitask.py` | ★ 단일 백본 멀티헤드 + multi-SNR (P1 확정 5d+SNR / α=0.7) |
| | `train_lora_multitask_ml.py` | multi-label 변형 (5e) |
| | `train_lora_multisnr.py` / `train_lora.py` / `train_lora_mixed.py` / `train_lora_multiclass.py` | 이진③ / 이진② / PTB-XL 혼합(5c) / 다중분류 단독(5b) |
| | `train_baseline.py` / `train_pilot_ptbxl.py` | 베이스라인 선형 프로빙 / 파일럿 |
| **증강** | `multisnr.py` | multi-SNR 모션 노이즈 주입 모듈 |
| **데이터 다운로드** | `download_cpsc2018.py`·`download_ptbxl.py`·`download_nstdb.py`·`download_cachet.py`·`download_external_dbs.py` | CPSC/PTB-XL/NSTDB/CACHET/(INCART·STAFF-III·LTST) |
| **전처리** | `preprocess_cpsc2018.py`(이진)·`preprocess_cpsc2018_mc.py`(5-class)·`preprocess_ptbxl.py`·`preprocess_sph.py`·`preprocess_cachet.py`·`preprocess_incart.py`·`preprocess_ltst.py`·`preprocess_staffiii.py`·`build_cpsc_sph.py` | raw → `data/processed/.../{train,val,test}/{signals,labels,...}.npy` |
| **평가** | `eval_multitask_test.py`·`eval_multitask_snr_ext.py`·`eval_snr_curve.py`·`eval_external.py`·`eval_mc_fair_compare.py`·`ablation_a1_rlm_leads.py`·`ablation_nlead_curve.py` | 테스트·SNR곡선·외부검증·공정비교·N-lead/RLM ablation |
| **검증/진단** | `stage0_spine.py`(단일리드 3축 검증)·`verify_env.py`·`verify_lora_attention.py`·`preflight_2d_features_only.py`·`diag_ltst_inversion.py`·`analyze_*.py` | 환경·LoRA·0-fill forward·진단 |

> `stage0_spine.py`는 공유 추론/검증 프레임워크: `LoRALinear`, `load_models(dev)`,
> `run(...)`, `single_lead(sig12)`(slot 1=II만 남기고 0-fill), `sens_spec(y,p,thr)`를 제공하며
> CACHET/NSTDB/PTT-PPG 3축 단일리드 검증에 쓰인다. 순수 유틸은 `tests/test_p1.py`가 스모크 테스트.

---

## 7. 데이터 흐름 (요약)

```
[Raw]  PhysioNet/figshare/HuggingFace
  ├ CPSC2018 ─ download_cpsc2018 ─┬ preprocess_cpsc2018      → data/processed/cpsc2018/        (이진)
  │                               └ preprocess_cpsc2018_mc   → data/processed/cpsc2018_mc(_ml)/ (5-class, ★학습)
  ├ PTB-XL  ─ download_ptbxl    ─ preprocess_ptbxl           → data/processed/ptbxl/    (patient-level split)
  ├ SPH     ─                    ─ preprocess_sph             → data/processed/sph/
  ├ NSTDB   ─ download_nstdb    ─ (런타임 MultiSNRNoise가 직접 로드)  data/raw/nstdb/{bw,em,ma}
  └ 외부검증 ─ download_cachet / download_external_dbs ─ preprocess_{cachet,incart,ltst,staffiii} → data/processed/{...}/ (split 없음, inference-only)

[Train]  train_lora_multitask.py
  data/processed/cpsc2018_mc/{train,val,test}  +  NSTDB(multi-SNR)
  → ECG-FM(frozen) + LoRA + BinaryHead + MulticlassHead
  → outputs/lora_multitask_snr_a07/lora_multitask_snr_best.pt  (α=0.7, epoch 18)

[Infer]  P1CardiacChannel.infer(signal_12x5000)
  → {emergency_score, cardiac_probs[5], benign_flag}   (※ embedding 미노출 — MODEL_INTERFACE.md 참조)
```

공통 처리 규격: 신호 float32 `(N, 12, 5000)` @500Hz, 라벨 int8 `(N,)`, split seed=42(70/15/15).
CPSC는 record-level(patient ID 미제공), PTB-XL·SPH는 patient-level split. 상세는 `REPRODUCIBILITY.md`.

---

## 8. 검증되지 않은/주의 항목 (요약)

- LoRA 주입 정준 경로(런타임 vs 라이브러리 패치) — §3.
- `.yaml` 사이드카 차원 stale(1024/24 vs 실제 768/12) — §2.
- 추론 인터페이스가 `embedding[768]`을 노출하지 않음(내부 계산 후 폐기) — `MODEL_INTERFACE.md`·`HANDOFF_ISSUES.md`.
- 멀티태스크(5f)의 INCART 외부 일반화 역전 — `HANDOFF_ISSUES.md` P0.

각 항목의 actionable 내용은 `HANDOFF_ISSUES.md` 참조.
