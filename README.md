# ECG-FM 기반 웨어러블 응급 심장 이상 탐지 (Project 1)

## 개요

웨어러블 환경에서 ECG Foundation Model(ECG-FM)을 효율 적응하여 **심혈관 응급(급성 허혈·AF)**을 탐지하는 모듈.
3-프로젝트 로드맵의 첫 번째 단계: **P1 ECG 분석** → P2 멀티모달 융합(자이로·SpO2) → P3 XAI 설명.

정적 임상 12-lead로 사전학습된 ECG-FM을 **LoRA(0.33% 파라미터)** 로 적응하고,
**multi-SNR 모션 증강 + RLM 가변 lead 마스킹** 으로 웨어러블 노이즈·lead 부족에 강건하게 만든다.
단일 백본에서 **이진 응급 점수 + 5-class 심장 분류 + 신호 신뢰도** 를 동시 출력하여 P2 융합 모듈로 전달한다.

---

## 핵심 기여

| 기여 | 내용 |
|---|---|
| **파라미터 효율 전이** | ECG-FM(90.9M) frozen + LoRA rank=8 — 전체의 0.33%(~295K)만 학습 |
| **모션 강건성** | NSTDB 기반 SNR {24,18,12,6,0}dB 합성 증강 → −6dB에서 AUROC +4.3%p 유지 |
| **N-lead 강건성** | RLM(Random Lead Masking p=0.5) → 1-lead 입력에서도 AUROC 0.9408 유지 |
| **단일 백본 다중 출력** | 이진 헤드 + 5-class 헤드 공유 (forward 1회, embedding 1개) — P2 인터페이스 정합 |
| **모달리티 신뢰도 게이트** | ECG-FM frozen + Linear → 연속 신뢰도(use/mask/alert) — P2 ECG 가중치 공급 |
| **적용 범위 경계 정의** | inter-patient 학습의 도메인 일반화 범위를 4개 외부 DB로 실증 |

---

## 아키텍처

```
ECG 입력 (12-lead · 500Hz · 10s · N-lead 0-fill)
          │
          ▼
  ECG-FM (Wav2Vec2CMSC, 90.9M, frozen)
  + LoRA (rank=8, α=16, q/v_proj, 295K params)
  [학습 시: RLM p=0.5 + multi-SNR 0~24dB]
          │
          ├── BinaryHead (768→1) → emergency_score
          ├── MCHead     (768→5) → cardiac_probs[5]
          └── mean-pool  z[768]  → embedding
          │
  (병렬) ECG-FM frozen → Linear(768→1) → reliability → use/mask/alert
          │
          ▼
  P1 Output Bundle → Project 2 (멀티모달 융합)
```

**P1 확정 모델**: `lora_multitask_snr_a07` (BCE 가중치 α=0.7, epoch 18)

---

## 결과 요약

### P1 확정 모델 (단일 백본 멀티태스크, α=0.7)

| 출력 | 지표 | 값 |
|---|---|---|
| 이진 응급 | AUROC | **0.9139** |
| 이진 응급 | Sens@95%Sp | 0.7072 |
| 5-class | Macro-F1 | **0.6858** |
| AF | AUROC | 0.9671 |
| 급성허혈(STD/STE) | AUROC | 0.9066 |
| 전도장애 | AUROC | 0.9577 |
| NSR | AUROC | 0.9304 |
| 이소성(PAC/PVC) | AUROC | 0.8634 |
| 신호품질 게이트 | AUROC | 0.8406 |

### 단일 이진 모델 참조 (③ LoRA+RLM+multi-SNR)

| 지표 | 값 |
|---|---|
| AUROC | 0.9463 |
| Sens@95%Sp | 0.7620 |
| 1-lead AUROC | 0.9408 |
| −6dB vs clean | +0.043 AUROC |

> 멀티태스크 모델(0.9139)과 단일 이진 모델(0.9463)의 −0.033 차이는
> "단일 forward로 이진+5-class+임베딩 동시 출력"에 대한 트레이드오프이며 의도된 설계 결정.

### 외부 도메인 일반화 (③ 기준)

| 데이터셋 | AUROC | 비고 |
|---|---|---|
| CACHET-CADB (웨어러블 AF) | **0.844** | frozen ECG-FM 0.847로 이미 강력 — 도메인 이동 없음 |
| INCART (병원 Holter, 12-lead) | 0.710 | frozen 대비 +13.6%p — multi-SNR 기여 |
| STAFF-III (장기 ST) | 0.517 | 라벨 체계 미정합 (scope boundary) |
| LTST (허혈 에피소드) | 0.407 | intra-patient 구분 필요 (scope boundary) |

> STAFF-III·LTST는 모델 한계가 아닌 **태스크 구조 불일치** (inter-patient 모델의 적용 범위 외)로 진단.
> 근거: `records/05_open_issues.md §1`

---

## P1 → P2 인터페이스

```json
{
  "cardiac_probs":   [NSR, AF, Ischemia, Conduction, Ectopic],
  "emergency_score": 0.91,
  "embedding":       [768개 float],
  "reliability":     0.18,
  "gate_tier":       "use",
  "physio":          {"hr_bpm": 73.2, "rhythm_regularity": 0.91},
  "model_version":   "lora_multitask_snr_a07_e18"
}
```

게이트 임계값: `use` < 0.2155 ≤ `mask` < 0.4753 ≤ `alert`
상세 명세: [records/00_research_plan.md §5](records/00_research_plan.md)

---

## 진행 단계

- [x] 환경 셋업 (RTX 3060 12GB, Python 3.9, PyTorch 2.1.2+cu118)
- [x] ECG-FM API Pre-flight (features_only=True, mean pooling PASS)
- [x] 데이터 확보 (CPSC2018 · PTB-XL · NSTDB · PhysioNet2011 · 외부검증 4종)
- [x] 전처리 파이프라인 (500Hz · 10s · 12-lead · patient-level split)
- [x] 베이스라인 Linear Probing (AUROC 0.9477, seed=42)
- [x] LoRA + RLM fine-tuning (AUROC 0.9477, F1 +0.018)
- [x] multi-SNR 증강 통합 (−6dB +4.3%p AUROC, Sens@95Sp +16.4%p)
- [x] SNR 저하 곡선 평가 (강건성 정량화)
- [x] N-lead 강건성 ablation (1-lead까지 AUROC 0.94)
- [x] 신호품질 게이트 (PhysioNet 2011, AUROC 0.8406, 3단 임계값 산출)
- [x] 외부검증 4종 (CACHET · INCART · STAFF-III · LTST)
- [x] PTB-XL 혼합 학습 — 외부 척도 확장 검증
- [x] 단일 백본 멀티태스크 설계 (Path B: BinaryHead + MCHead)
- [x] α=0.7 손실 가중치 최적화 — P1 확정 모델
- [x] P1→P2 인터페이스 명세 확정 (단계 10)

---

## 데이터 분할 및 누출 방지

- **CPSC 2018**: record-level random split (seed=42, 70/15/15) — patient ID 미제공
- **PTB-XL**: patient-level split (seed=42, 70/15/15) — patient_id 제공
- **외부검증 4종**: 학습 미사용 inference-only held-out
- **ECG-FM 사전학습과 비중복**: STAFF-III · LTST · CACHET · NSTDB는 PhysioNet 2021 미포함

---

## 참조

- ECG-FM: McKeen et al., JAMIA Open 2025 (arXiv 2408.05178)
- LoRA: Hu et al., ICLR 2022
- RLM: Oh et al., CHIL 2022 (arXiv 2203.06889)
- Multi-SNR: MDPI Sensors 26(4):1135, 2026
- Signal Quality: Mondal 2025 (Biomed Signal Process Control)
