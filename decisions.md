# ECG-FM Emergency Detection — 프로젝트 기록 인덱스

> GitHub: `WidU_P1_LoRA-PEFT_Foundation-Model_Adaptation`
> 데이터·출력은 repo-로컬 `data/`·`outputs/` (대용량이라 미추적)
> 최종 갱신: 2026-06-03 | **Project 1 완료 (모델·연구 안정화)** · RLM 역할 재점검 보강(⑥-c)
>
> cardiac 채널은 웨어러블 단일리드 적용까지 검증 완료(단계 11). 모델은 동결 상태이며,
> 추가 변경(예: 온디바이스 경량화를 위한 증류)이 필요할 때만 재방문한다.

---

## records/ 폴더 구조

| 파일 | 내용 |
|---|---|
| [records/00_research_plan.md](records/00_research_plan.md) | 연구계획서 — 3-프로젝트 로드맵, P1 다중출력 설계, **§5 P1→P2 인터페이스 명세 (확정)** |
| [records/01_design_decisions.md](records/01_design_decisions.md) | 설계 결정 — 아키텍처 선택 근거, LoRA·RLM·다중분류 설계, **전처리 방법론 7가지**, 멀티태스크 Path B 채택, α=0.7 메커니즘 |
| [records/02_training_logs.md](records/02_training_logs.md) | 전 에폭 훈련 로그 — ①②③ 이진, 5b 다중분류, 5d/5d+SNR 멀티태스크, 5e multi-label, **⑧ 5f α=0.7 (P1 확정)** |
| [records/03_eval_results.md](records/03_eval_results.md) | 평가 결과 — 테스트 지표, SNR 곡선, 외부검증 4종, 5b/5d/5e 비교, **단계 5f P1 최종 수치** |
| [records/04_run_history.md](records/04_run_history.md) | 실행 기록 — 환경 설정, 다운로드·전처리 이력 (LTST OOM 수정 포함) |
| [records/05_open_issues.md](records/05_open_issues.md) | 이슈 트래커 — LTST/STAFF-III 범위 경계 분석, P1 확정, Stage 10, 이소성 표현 천장 |

---

## 완료된 전체 단계

| 단계 | 내용 | 핵심 결과 |
|---|---|---|
| 1 | 환경 설정 | RTX 3060 12GB, Python 3.9, PyTorch 2.1.2+cu118 |
| Pre-flight | ECG-FM API 확정 | features_only=True, mean pool (B,312,768)→(B,768) PASS |
| 3 | CPSC 2018 전처리 | train=2217 / val=474 / test=474 (이진), mc: 4357/933/936 |
| 4 | 베이스라인 Linear Probing | AUROC=0.9477 (seed=42, epoch 25) |
| ② | LoRA + RLM | AUROC=0.9477, F1 +0.018 vs ① |
| ③ | LoRA + RLM + multi-SNR | AUROC=0.9463, **Sens@95Sp=0.7620** |
| 5b | 심장 다중분류 단독 (5-class) | Macro-F1=0.6762, AF AUROC=0.9756 — 성능 기준선 |
| 5c | PTB-XL 혼합 이진 학습 | CPSC AUROC=0.9714 — 외부 척도 확장 검증 |
| 6 | SNR 저하 곡선 | −6dB: ③−②= +0.043 AUROC, +16.4%p Sens — 강건성 정량화 |
| Ablation A1 | N-lead 강건성 | 1-lead까지 AUROC 0.9408 유지 |
| 9 외부검증 | CACHET-CADB | ③ AUROC=0.844 (frozen ①=0.847 대비 동등) |
| 9 외부검증 | INCART (병원 Holter) | ③ AUROC=0.710 (+13.6%p vs ①=0.574) |
| 9 외부검증 | STAFF-III / LTST | 0.517 / 0.407 — **범위 경계** (inter-patient 모델 적용 외) |
| 5d+SNR | 단일 백본 멀티태스크 + multi-SNR | Path B 채택 — AUROC=0.9134, Macro-F1=0.6834 |
| 5e | multi-label 데이터 결함 수정 | 이소성 ~0.82 천장 확인 — ECG-FM 표현 병목으로 결론 |
| **5f** | **α=0.7 손실 가중치 최적화** | **BinAUROC=0.9139, Macro-F1=0.6858 — P1 확정** |
| 10 | P1→P2 인터페이스 명세 | JSON/npz 규약, 출력 5종 계약, P2 소비 규약 확정 |
| 감사 | P1 심층 재검토 (코드·데이터) | 절단 64.7%(약클래스 오진 가능)·이진vs멀티 split교란·멀티토큰 기각(cos 0.999) (records/05) |
| **11** | **웨어러블 단일리드 적용 연구** | CACHET 단일리드 AF 0.99(낙관적), 온디바이스 추론 CPU 16x (records/03) |
| 재점검 | RLM 역할 — multi-SNR 레시피 내 분리 + 리드×SNR 그리드 | no-RLM 급락 없음(1-lead 0.940); RLM 마진 단일리드+모션 집중(1-lead/6dB +0.017 AUROC·+0.096 Sens), 주동력=백본+multi-SNR → RLM은 보강 (records/03 ⑥-c) |

---

## P1 확정 모델 — 빠른 참조

### 단일 백본 멀티태스크 α=0.7 (★ P1 공식 모델)

> 체크포인트: `outputs/lora_multitask_snr_a07/lora_multitask_snr_best.pt` (epoch 18)

| 출력 | 지표 | 값 |
|---|---|---|
| 이진 응급 | AUROC | **0.9139** |
| 이진 응급 | Sens@95%Sp | 0.7072 |
| 이진 응급 | F1@0.5 | 0.7887 |
| 다중분류 | Macro-F1 | **0.6858** |
| AF | AUROC | 0.9671 |
| 급성허혈(STD/STE) | AUROC | 0.9066 |
| 전도장애 | AUROC | 0.9577 |
| NSR | AUROC | 0.9304 |
| 이소성(PAC/PVC) | AUROC | 0.8634 |

### 이진 단독 모델 참조 (③, 강건성 서사 기준)

| 모델 | AUROC | F1 | Sens@95%Sp |
|---|---|---|---|
| ① 베이스라인 (frozen+linear) | 0.9477 | 0.9104 | 0.7592 |
| ② LoRA+RLM | 0.9477 | 0.9284 | 0.7564 |
| ③ LoRA+RLM+multi-SNR | **0.9463** | 0.9155 | **0.7620** |

> ③는 단일 이진 최고 성능 모델. P1 확정(5f)과의 −0.033 AUROC 차이는
> **단일 백본 멀티태스크 트레이드오프** (forward 1회·embedding 1개·5-class 동시 출력).

### 외부검증 (③ 기준)

| DB | AUROC | 비고 |
|---|---|---|
| CACHET-CADB (웨어러블 AF) | **0.844** | 사전학습 ECG-FM 이미 강력 — 추가 전이 효과 제한 |
| INCART (병원 Holter 12-lead) | 0.710 | multi-SNR +13.6%p — 증강의 도메인 일반화 기여 |
| STAFF-III (장기 ST 기록) | 0.517 | 범위 경계 — 라벨 체계 미정합 |
| LTST (허혈 에피소드) | 0.407 | 범위 경계 — intra-patient 구분 불가 (설계 외 태스크) |

### 웨어러블 cardiac 채널 (단계 11, 단일리드 적용 연구)

| 항목 | 값 |
|---|---|
| AF 전이 (CACHET 단일리드) | AUROC 0.9935 (윈도우단위·낙관적) |
| 온디바이스 추론 (연산) | ECG-FM 90.9M, CPU 637ms/window = 16x 실시간 (모바일 변환·전력은 미검증) |
| 검출 실증 | **AF만**(CACHET) — 낙상·저산소·2리드는 실세계 검증 연구 |
| 산출물 | `p1_cardiac_channel.py`, `p1_cardiac_logging_adapter.py` |

---

## 체크포인트 위치

```
outputs/
├── baseline/baseline_best.pt                          epoch=25, AUROC=0.9477 (이진 ①)
├── lora/lora_best.pt                                  epoch=21, AUROC=0.9477 (이진 ②)
├── lora_multisnr/lora_multisnr_best.pt               epoch=30, AUROC=0.9463 (이진 ③, 강건성 기준)
├── lora_no_rlm/lora_best.pt                           epoch=16, AUROC=0.9436 (ablation A1, ② clean 레시피)
├── lora_no_rlm/lora_multisnr_best.pt                  epoch=17, test=0.9417 (⑥-c, ③ multi-SNR 레시피 no-RLM)
├── lora_mc/lora_mc_best.pt                            epoch=27, MacroF1=0.6762 (5b 단독 MC)
├── lora_mixed/lora_mixed_best.pt                      epoch=7,  AUROC=0.9714 (5c PTB-XL 혼합)
├── lora_multitask/lora_multitask_best.pt              epoch=27, composite=0.8046 (5d)
├── lora_multitask_snr/lora_multitask_snr_best.pt     epoch=22, composite=0.8046 (5d+SNR, α=0.5)
├── lora_multitask_snr_a07/lora_multitask_snr_best.pt epoch=18, composite=0.8051 (★ P1 확정, α=0.7)
└── lora_multitask_ml/lora_multitask_ml_best.pt       epoch=13, composite=0.9023 (5e multi-label)
```

---

> 상세 기록: `records/` 폴더 참조.
> 로그 규칙: 새 훈련 완료 시 전 에폭 로그를 `records/02_training_logs.md`에 추가 + 커밋.
