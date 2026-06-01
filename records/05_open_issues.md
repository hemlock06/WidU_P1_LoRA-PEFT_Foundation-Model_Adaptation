# 이슈 트래커 (Open Issues & Decisions)

> Project 1 완료 기준일: 2026-05-29
> 미결 = Project 2 착수 전 선택적 개선 과제.

---

## 미결 과제 (P2 착수 전 선택적)

### 게이트 모델 LoRA 파인튜닝 (3b)

- **상황**: 현 게이트 = ECG-FM frozen + Linear probe, AUROC=0.8406.
- **병목 확인**: 7b 분석에서 임계값 튜닝은 zero-sum (양호 use↑ = 불량 오인↑). 게이트 AUROC 자체가 병목.
- **개선 여지**: 이진 헤드 전례(linear 0.9398→LoRA 0.9477)로 보아 게이트도 LoRA 파인튜닝 시 0.84 상회 예상.
- **조치**: `scripts/train_gate_lora.py` 신규 + PhysioNet 2011 재학습 → 7b 임계값 재산출.
- **우선순위**: P1 완료 후 선택. P2 멀티모달 융합에서 ECG 가중치 품질에 직결.

---

## 완료된 이슈 기록 (결정 근거 참조용)

### LTST·STAFF-III 범위 경계 정의 (완료, 2026-05-26)

**진단**: inter-patient 모델의 적용 범위 외 태스크.
- LTST AUROC 0.407 역전: 동일 환자의 허혈 에피소드 중 vs 비에피소드 구분 (intra-patient 태스크) — 본 모델 설계 외.
- STAFF-III AUROC 0.517: 라벨 체계 미정합 (balloon inflation 기준 vs 모델의 CPSC 기준).
- **결론**: 모델 한계 아님. inter-patient 학습 모델의 태스크 범위 경계로 명시. (`records/03 §⑦`)

### P1 아키텍처 — 단일 백본 멀티태스크 (완료, 2026-05-29)

**결정**: 단일 백본 (ECG-FM + LoRA 1개) + BinaryHead + MCHead. 하이브리드(LoRA 2개) 탈락.
- 하이브리드 탈락 근거: forward 2회·임베딩 2개·P2 인터페이스 훼손 — 성능 동급에서 비용만 2배.
- 손실 가중치 α=0.7 확정: BCE 그래디언트 지배 → z "응급 축" 정렬 → MacroF1 향상 (그래디언트 간섭 해소).
- 상세: `records/01_design_decisions.md §α=0.7`, `records/03 §5f`

### 게이트 3단 임계값 (완료, 2026-05-29)

**결정**: t_mask=0.2155 (val 양호 p75), t_alert=0.4753 (val 양호 p90, spec≈0.90), 결정성 고정.
- 5-28 초기 산출 교체: (1) 비결정 추론 → 재현불가, (2) Youden J 피레토 지배점 교체.
- trade-off: t_mask↑는 양호 use(+6%p) ↔ 불량 오인(+4개) — zero-sum. 현 운영점 유지.
- 상세: `records/03 §7b`, `outputs/gate/gate_thresholds.npz`

### P1→P2 인터페이스 명세 Stage 10 (완료, 2026-05-29)

**결정**: `records/00_research_plan.md §5` 완전 확정.
- 출력 5종: cardiac_probs[5] · emergency_score · embedding[768] · reliability · gate_tier
- 직렬화: 실시간=JSON, 배치=npz
- gate 임계값: t_mask=0.2155, t_alert=0.4753
- physio 계산: HR(R-R 기반), rhythm_regularity(sdNN 기반) — `physio_features.py` 미구현 (future)

### 다중분류 이소성 표현 천장 (완료, 2026-05-29)

**진단**: ECG-FM mean-pool 표현에서 이소성(PAC/PVC) ~0.82가 상한.
- 데이터 결함 수정(164884008 복구, train 372→909): 이소성 AUROC 변화 없음 → 데이터 부족 가설 반증.
- pooling 방식 변경(max/std): 효과 없음 → 풀링 병목 가설 반증.
- **결론**: ECG-FM frozen 표현 자체의 이소성 구분력 한계. P2 융합(타이밍·자이로)에서 보완 예정.
- 미션 영향 없음: 이소성은 비응급(labels_bin=0) → 응급 판정 무관.
- 상세: `records/01 §다중분류 약클래스`, `records/03 §5e`

### PTB-XL 혼합 학습 (완료, 2026-05-27)

**결론**: 이진 분류 외부 척도 확장 자료로 활용 (CPSC test AUROC=0.9714).
- 외부 4종 변화 ±0.02 이내 → P1 헤드라인 개선 목적엔 효익 제한적.
- 시행착오 4건 (LR·비율·warm_start 조합): `records/04_run_history.md` 상세.
- 최종 안정 조합: `--lr 1e-5 --ptbxl_ratio 0.3 --ckpt_start lora_multisnr_best.pt`

### 공유 잠재공간 오토인코더 (기각, 2026-05-28)

**기각**: 논문 자체가 오토인코더 접근을 폐기하고 단순 baseline이 더 낫다고 결론.
- 저채널 강건성은 RLM+N-lead ablation이 이미 우위 (1-lead AUROC 0.94).
- ECG-FM 백본과 아키텍처 충돌 (raw 5000샘플 입력 요구 vs 잠재벡터 먹이기 불가).
- 상세: `records/05_open_issues.md §R1` (구 버전) → `records/01_design_decisions.md`로 이관.

### P1 심층 재검토 — 코드·데이터 전수 감사 (2026-05-30)

P1을 코드·데이터 레벨로 심층 재감사. 4가지 발견(일부 자기수정):

1. **★ 첫-10초 절단 — 약클래스 오진 가능성** (전처리 함정, **미해결**):
   - `preprocess_cpsc2018*.py`가 10초 초과(**64.7%**)를 앞 5000샘플만 쓰고 폐기(평균 37%).
   - 클래스별 폐기율: **이소성 47.6%(최대)** · 허혈 31.9% vs AF 39%(가장 덜 잘리고 가장 강함).
   - → 이소성 "표현 천장(~0.82)" 결론은 **윈도잉을 한 번도 의심 안 함**. 사건이 첫 10초에
     없으면 라벨 노이즈 = "윈도우 천장"일 수 있음. (단 이소성=benign, 미션 영향 작음)
   - 개선책: 전체 레코드 다중윈도우(MIL) 집계 — 미실행, 향후 약클래스 개선 레버.

2. **③ vs 5f 비교는 다른 테스트셋** (측정 교란, 문서화):
   - 이진(cpsc2018) test와 mc(cpsc2018_mc) test는 **14%만 중복**, 68%가 상호 train.
   - "③ 0.946 vs 5f 0.914"의 −0.033은 다른 레코드·다른 난이도(mc는 전도/이소성 음성 포함)에서
     측정 → "멀티태스크 희석" 진단은 부분 교란. **단일백본 선택은 오히려 더 정당**.

3. **멀티토큰/풀링 — 기각 (자기수정)**:
   - ECG-FM 312 시간토큰 인접 cos **0.999**(거의 stationary) → 멀티토큰 export 무의미.
   - 이소성 풀링 비교: max 0.796 / std 0.820 < **mean 0.837** → mean-pool 무죄(잔차 회복 불가).

4. **정규화·외부스케일 — 정상 확인**: 체크포인트 `cfg.normalize=False`(raw mV 정합). INCART
   전체-레코드 AFIB 라벨·리드순서 미검증은 외부검증 한계로 기존 문서화.

### CACHET-CADB 학습형 개인화 — 데이터 타당성 점검 (NO-GO, 2026-06-01)

**가설**: 개인 ECG baseline을 학습해 이탈을 이상으로 검출하는 within-subject 개인화가 AF 검출을
개선하는가. 성립 조건 = 다수 subject가 사람당 충분한 AF 이벤트 보유(시간분할 적응→평가).

**방법**: CACHET-CADB 종단 주석(per-subject `annotation.csv`: Start,End,Class) 집계. Class 매핑은
데이터 descriptor(Frontiers Cardiovasc. Med. 2022, Table 7): **1=AF, 2=NSR, 3=Noise, 4=Others**.
16GB 신호는 미해제, 주석 CSV만 파싱(`scripts/analyze_cachet_personalization.py`).

**측정**: 총 1602 주석(논문 수치 일치). 전역 분포 AF 747 / NSR 615 / Noise 221 / Others 19.
→ **AF(이상) ≥20건 보유 = 6/23 subject뿐**(AF≥1 = 11명, 논문 "11 patients had AF" 일치). 18명은
AF 사실상 없음. 범례-무관 시 충분해 보였던 다수-event 클래스는 NSR(정상 baseline)이었음.

**판정 (NO-GO)**: 이상(AF)이 소수 subject에 편중 → within-subject 시간분할 적응·평가가 다수에서
불성립. 게다가 AF 모집단 검출이 이미 높은 수준(records/03)이라 편중 코호트에서 개인화의 한계이득
입증 곤란. → **학습형 개인화 미진입, 실세계 종단 데이터 확보 시 재개 대상으로 보류.** 이상 희소성은
건강 모니터링 데이터의 본질 — per-subject 편중이면 개인화 학습 불가가 정상 결론.

---

## 주의사항

- **eval 비결정성**: cuDNN 비결정 연산으로 실행 간 AUROC ±~0.002 변동.
  4자리 정밀 재현은 불가. 결론들은 변동폭(±0.002)보다 훨씬 큰 차이(+0.02~+0.04)에 근거.
