# 연구계획서 (Research Plan)

> 본 문서는 3-프로젝트 로드맵에서 **Project 1(ECG 모듈)**의 산출물 설계 및 단계별 계획을 기술한 마스터 계획서다.
> Project 1 출력: 심장 다중분류 + 이진 응급 점수 + ECG-FM 임베딩 + 신호 신뢰도

---

## 0. 전체 비전 — 3-프로젝트 로드맵

| 프로젝트 | 역할 | 입력 | 출력 |
|---|---|---|---|
| **Project 1 (현재)** | 심전도 분석 | 12/N-lead ECG | 심장 다중분류 + 이진 응급 + 임베딩 + 신뢰도 |
| **Project 2 (차기)** | 멀티모달 융합 | P1 출력 + 자이로(스마트폰) + SpO2(스마트워치) | 응급 **원인** 다면 분류 (심혈관/외부충격/저산소 등) |
| **Project 3 (최종)** | XAI 설명 | P2 결과 | 누구나 이해 가능한 근거 설명 |

### 모달리티 역할 분담 (핵심 설계 원칙: separation of concerns)

각 모달리티는 **자신이 1차 관측 가능한 것만** 분류한다. 침범하지 않는다.

| 응급 원인 | 1차 관측 모달리티 | Project |
|---|---|---|
| 심혈관계 (AF·급성허혈 등) | **ECG** | P1이 분류 |
| 외부충격 (낙상·충격) | **자이로** | P2가 분류 |
| 저산소 | **SpO2** | P2가 분류 |
| 일상/운동 구분 | 자이로(활동) + ECG(심박) | P2가 융합 판정 |

> **5분류(정상1/정상2/심혈관계/외부충격/산소저하)는 Project 2(융합 시스템)의 최종 출력 taxonomy로 설계됐다.**
> Project 1은 ECG로 비관측되는 외부충격·산소저하를 분류하지 않는다 (라벨 노이즈·모달리티 충돌 방지).
> Project 1은 그중 "심장 채널"을 최대한 풍부하게 제공하는 데 집중한다.

---

## 1. Project 1 산출물 설계 (Project 2와의 인터페이스 계약)

Project 1은 "이상/정상 1비트"가 아니라 융합·XAI가 활용할 **다중 출력 묶음**을 제공한다.

| # | 산출물 | 형태 | 다운스트림 용도 |
|---|---|---|---|
| 1 | **심장 다중분류 확률벡터** | softmax (5-class) | "어떤 심장 이벤트인가" 구체 원인 → XAI 설명 |
| 2 | **이진 응급 점수** | sigmoid (1) | 이상/정상 직관 전달 (헤드라인·강건성 지표) |
| 3 | **ECG-FM 임베딩** | float (768) | 융합 모델의 풍부한 잠재 입력 |
| 4 | **신호 신뢰도 점수** | float (0~1, 게이트) | ECG 채널 융합 가중치 (use/mask/alert) |
| 5 | 파생 생리지표 | 심박·리듬 규칙성 | 자이로·SpO2와 교차맥락 (심박↑+움직임↑=운동) |

### 심장 다중분류 taxonomy (CPSC 2018 native 라벨 재활용 — 신규 수집 0)

| 심장 클래스 | CPSC 원본 | 임상 중증도 | 이진 매핑 |
|---|---|---|---|
| 정상 (NSR) | Normal | — | 정상(0) |
| 심방세동 (AF) | AF | 위험 부정맥 | 응급(1) |
| 급성허혈 ST변화 | STD + STE | 응급 | 응급(1) |
| 전도장애 | I-AVB + LBBB + RBBB | 대부분 양성 | 정상(0)* |
| 이소성 박동 | PAC + PVC | 대부분 양성 | 정상(0)* |

\* 이진 헤드에서 전도장애·이소성의 매핑은 학습 시 확정 (현 이진 모델은 이 5클래스를 제외했으나, 다중분류에서는 고유 클래스로 복귀).
- 이진 헤드와 다중분류 헤드는 **동일 백본 위에 병행**. 기존 강건성 서사(SNR곡선·N-lead) 보존.

---

## 2. 단계별 계획 (현황 반영)

| 단계 | 내용 | 상태 | 비고 |
|---|---|---|---|
| 1 | 환경 셋업 + Pre-flight 검증 | 완료 | RTX 3060, ECG-FM 0-fill PASS |
| 2 | 데이터 확보 (CPSC/NSTDB/외부 4종) | 완료 | LTST 포함 86/86 레코드 완료 |
| 3 | 전처리 (500Hz/10s/12-lead) | 완료 | train 2217/val 474/test 474 |
| 4 | 베이스라인 Linear Probing | 완료 | AUROC 0.9477 (seed=42) |
| 5 | LoRA + RLM + multi-SNR | 완료 | −6dB +4.3%p AUROC |
| 5b | **심장 다중분류 헤드 추가** | 완료 (2026-05-26) | Macro-F1=0.6762, 이진AUROC=0.9263, AF AUROC=0.9756 |
| 6 | SNR 저하 곡선 | 완료 | 모션 강건성 입증 |
| 7 | 신호품질 게이트 | 완료 | ECG-FM+Linear AUROC 0.84 |
| 7b | **게이트 → 연속 신뢰도 + 3단 임계값** | 신규 | use/mask/alert 임계값 레이어 (재구축 불요) |
| 8 | 외부검증 (CACHET/INCART/STAFF-III/LTST) | 완료 | LTST AUROC 역전 패턴 — 한계 문서화 |
| 9 | N-lead 강건성 ablation | 완료 | 1-lead까지 AUROC 0.94 |
| 10 | **다중분류 평가 + 인터페이스 명세** | 신규 | per-class 지표 완료, P2 인터페이스 포맷 확정 필요 |

---

## 3. 게이트 설계 — ECG 모달리티 신뢰도 공급자

**융합 시스템에서 게이트 = "ECG 모달리티 신뢰도 공급자"** 로 설계된다.
단순 품질 이진 필터를 넘어 P2의 모달리티 가중치 입력으로 기능한다.

| 게이트 출력 (연속 0~1 임계값) | 융합 동작 |
|---|---|
| **use** (신뢰도 高) | ECG 가중치 크게 |
| **mask** (일부 lead 불량) | 가용 lead만, 가중치 中 |
| **alert** (신뢰도 低) | 자이로·SpO2로 무게 이동 |

- 현 ECG-FM+Linear 게이트(AUROC 0.84)가 이미 **연속 확률**을 출력 → 신뢰도 점수 v1로 직접 재해석.
- per-lead 1D CNN 재구축 불요 (PhysioNet 2011 per-channel 라벨 부재 한계 회피).
- {use/mask/alert} 3단 임계값 레이어는 멀티모달 맥락에서 ECG 채널 가중치 분기 기준으로 활용된다.

---

## 4. 주요 설계 결정 근거

| 항목 | 초기 고려 사항 | 채택 설계 | 근거 |
|---|---|---|---|
| Project 1 출력 | 이상/정상 이진 단일 출력 | 심장 다중분류 + 이진 병행 + 신뢰도 | P2/P3 설명력 확보 — 단순 이진 판정보다 원인 정보가 융합·XAI에 필수 |
| 5분류 응급 원인 | ECG 단일 모달로 전부 분류 | **심장 계열만 P1, 나머지는 P2 융합 출력** | 외부충격·산소저하는 ECG로 관측 불가 — 모달리티 충돌·라벨 노이즈 방지 |
| 신호품질 게이트 출력 | per-lead 1D CNN + {use/mask/alert} 이진 출력 | ECG-FM+Linear 연속 신뢰도 점수 + 임계값 레이어 | per-channel 라벨 부재 + ECG-FM 백본 재사용 + P2 가중치 입력 역할 |
| 신호 입력 형식 | dECG(미분 신호) | 원신호 (ECG-FM mV 원시 입력) | ECG-FM 사전학습 입력과 일치 — 미분 시 표현 손상 |

---

## 5. Project 1 → Project 2 인터페이스 명세 (단계 10 확정, 2026-05-29)

> **P1 공식 모델 확정 (2026-05-29)**: `lora_multitask_snr_a07` — ECG-FM + LoRA(α=0.7·BCE) + multi-SNR.
> 단일 백본 유지 근거: (1) forward 1회 (2) embedding 단일성 (3) 배포 정합성.
> α=0.7 실험 결과: BinAUROC=0.9139, MacroF1=0.6858 — 5d+SNR(0.5) 대비 test 동점, val composite +0.0005 우위.
> 하이브리드(LoRA 2개·forward 2회) 탈락: 성능 동급이나 연산 2배·임베딩 2개 문제.

---

### 5-1. 출력 계약 (Output Contract)

```python
P1_output = {
    # ── 심장 진단 ──────────────────────────────────────────────────
    "cardiac_probs": List[float],   # 길이 5, softmax 합=1.0
    # 인덱스 순서: [NSR, AF, Ischemia, Conduction, Ectopic]
    # 0=NSR(정상동율동), 1=AF(심방세동), 2=Ischemia(STD/STE),
    # 3=Conduction(I-AVB/LBBB/RBBB), 4=Ectopic(PAC/PVC)

    # ── 이진 응급 ──────────────────────────────────────────────────
    "emergency_score": float,       # 0~1, sigmoid (AF+허혈 합산 확률)
    # 소스: lora_multitask_snr_a07 BinaryHead (단일 백본 확정)
    # AUROC=0.9139, Sens@95Sp=0.7072 (CPSC mc test, epoch 18)

    # ── ECG-FM 임베딩 ───────────────────────────────────────────────
    "embedding": List[float],       # 길이 768, L2 정규화 없음 (raw mean-pool)
    # 용도: P2 멀티모달 융합 입력, P3 XAI 유사도 검색

    # ── 신호 신뢰도 ────────────────────────────────────────────────
    "reliability": float,           # 0~1 (높을수록 ECG 불량)
    # 산출: gate_best.pt sigmoid 출력 (PhysioNet 2011 학습)
    # AUROC: val=0.765, test=0.833

    "gate_tier": str,               # "use" | "mask" | "alert"
    # 임계값 (gate_thresholds.npz, 결정성 고정 seed=42):
    #   use   : reliability < t_mask  (= 0.2155, val_양호_p75)
    #   mask  : t_mask ≤ reliability < t_alert
    #   alert : reliability ≥ t_alert (= 0.4753, val_양호_p90, spec≈0.90)

    # ── 파생 생리지표 ───────────────────────────────────────────────
    "physio": {
        "hr_bpm": float,            # 심박수 (bpm). 계산: 60 / mean(R-R intervals@500Hz)
        "rhythm_regularity": float, # 0~1, 1=완전 규칙 (sdNN 기반)
        # 계산: 1 - clip(sdNN / 200ms, 0, 1), sdNN = std(R-R intervals in ms)
        # R-peak 미검출 시: hr_bpm=null, rhythm_regularity=null
    },

    # ── 메타 ───────────────────────────────────────────────────────
    "model_version": str,           # 체크포인트 식별자: "lora_multitask_snr_a07_e18"
    "inference_ms":  float,         # 단일 레코드 추론 시간 (ms)
}
```

---

### 5-2. 직렬화 규약

**단일 추론 응답 (P1→P2 실시간)**: JSON

```json
{
  "cardiac_probs":    [0.02, 0.87, 0.08, 0.02, 0.01],
  "emergency_score":  0.91,
  "embedding":        [0.031, -0.142, ...],
  "reliability":      0.18,
  "gate_tier":        "use",
  "physio":           {"hr_bpm": 73.2, "rhythm_regularity": 0.91},
  "model_version":    "lora_multitask_snr_a07_e22",
  "inference_ms":     47.3
}
```

**배치 저장 (평가·로깅)**: NumPy `.npz`

```python
np.savez("p1_batch.npz",
    cardiac_probs   = arr_float32_Nx5,
    emergency_score = arr_float32_N,
    embedding       = arr_float32_Nx768,
    reliability     = arr_float32_N,
    gate_tier       = arr_str_N,           # dtype='U8'
    hr_bpm          = arr_float32_N,
    rhythm_regularity = arr_float32_N,
    record_ids      = arr_str_N,
    model_version   = np.array(["lora_multitask_snr_a07_e22"]),
)
```

---

### 5-3. P2 소비 계약 (Consumption Contract)

P2(멀티모달 융합 모델)가 P1 출력을 소비할 때 지켜야 할 규약:

| P1 필드 | P2 사용 방식 | gate_tier 조건 |
|---|---|---|
| `emergency_score` | ECG 응급 채널 (헤드라인) | gate_tier = "use"/"mask" 시 사용, "alert" 시 무시 |
| `cardiac_probs` | 심장 원인 분류 (XAI 설명) | gate_tier = "use" 시 전체 사용, "mask" 시 참고만 |
| `embedding` | 융합 입력 (768-dim 잠재 벡터) | gate_tier 무관 — 모달리티 가중치로 보정 |
| `reliability` | ECG 모달리티 가중치 `w_ecg = 1 - reliability` | 항상 사용 (연속 가중치) |
| `gate_tier` | 이산 라우팅 분기 | "alert" → 자이로·SpO2로 무게 이동 |
| `physio.hr_bpm` | 자이로 활동량과 교차맥락 | null 허용 — null이면 다른 HR 소스로 대체 |

**ECG 모달리티 가중치 예시**:
```python
w_ecg = 1.0 - p1["reliability"]           # 0.0(불량) ~ 1.0(양호)
w_ecg *= 0.0 if p1["gate_tier"] == "alert" else 1.0  # alert 시 차단
fusion_input = w_ecg * p1["embedding"] + (1 - w_ecg) * other_modality_emb
```

---

### 5-4. 파생 생리지표 구현 메모 (단계 10 후속)

HR 및 리듬 규칙성은 별도 경량 R-peak 검출기(`scripts/physio_features.py`, 미구현)로 산출:

```python
# 의존: scipy.signal.find_peaks 또는 NeuroKit2(선택)
def compute_physio(ecg_12lead: np.ndarray, fs=500) -> dict:
    """
    ecg_12lead: (12, 5000) float32
    Lead II (index 1)를 기준 R-peak 검출.
    """
    lead_ii = ecg_12lead[1]   # Lead II
    peaks, _ = find_peaks(lead_ii, distance=int(0.3*fs), height=0)
    if len(peaks) < 2:
        return {"hr_bpm": None, "rhythm_regularity": None}
    rr_ms = np.diff(peaks) / fs * 1000          # ms
    hr_bpm = 60_000 / rr_ms.mean()
    sdnn   = rr_ms.std()
    regularity = float(np.clip(1 - sdnn / 200, 0, 1))
    return {"hr_bpm": round(float(hr_bpm), 1),
            "rhythm_regularity": round(regularity, 3)}
```

> 이 함수는 P1 추론기(`infer_p1.py`, 미구현)에서 모델 forward 이후 호출.
> R-peak 검출 실패 시 null 반환 — P2에서 null 처리 필수.

---

## 6. 구현 현황 — Phase 1 완료 자산과 Phase 2 진행 상태 (2026-05-28)

### 산출물 달성 현황

| 계획 (§1) | 현 자산 | 정합 여부 |
|---|---|---|
| **동일 백본** 위에 이진 + 다중분류 헤드 **병행** | 백본 3개 분리 (lora_multisnr / lora_mixed / lora_mc) | 불일치 |
| 다중분류 5-class softmax | `lora_mc_best.pt` — CPSC만 사용 | △ 부분 충족 |
| 이진 응급 score | `lora_multisnr_best.pt` (단일) 또는 `lora_mixed_best.pt` (혼합) | 충족 |
| ECG-FM 임베딩 (768) | 백본별로 추출 가능 | 충족 |
| 신호 신뢰도 (게이트) | `gate_best.pt` AUROC 0.8406 | 충족 |
| 파생 생리지표 (HR/리듬) | 미구현 | 단계 10에서 |

### Phase 1 자산과 Phase 2 목표 간 현황

PTB-XL은 Phase 1 이진 검증 단계에서 **외부 척도 확장 자료** 로 활용되었다(lora_mixed, CPSC test AUROC 0.9714). 이는 Phase 1 목적에 부합하며, Phase 2에서 PTB-XL SCP 코드(70+)를 5-class taxonomy(NSR/AF/허혈/전도/이소성)로 매핑하여 다중분류 헤드 학습 데이터로 통합하는 것이 다음 단계 과제이다.

### Phase 2 구현 경로 (상세 평가: `records/05_open_issues.md` 이슈 #6)

| 경로 | 설명 | 상태 |
|---|---|---|
| **A. 멀티헤드 단일 백본 (채택)** | `train_lora_multitask.py`: 공유 ECG-FM+LoRA → BinaryHead + MulticlassHead, BCE+CE 복합 손실 | **진행 중 (5d, epoch 7/30)** |
| **B. 현 자산 직렬 호출** | 이진·다중분류 백본 별도 forward | (미채택 — 단일 백본 설계 원칙 위배) |
