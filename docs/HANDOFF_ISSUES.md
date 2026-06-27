# HANDOFF_ISSUES — P0/P1/P2 actionable

> 통합·인수인계 시 처리해야 할 항목. 우선순위 정의:
> **P0** = 배포/통합 전 반드시 판단·조치 / **P1** = 품질·재현성 개선(통합과 병행 가능) /
> **P2** = P2 연결·후속 단계 작업.
> 모든 항목은 레포 코드·`records/` 기록에서 **검증된 사실**에 근거한다(추측 배제).
> 근거 위치를 각 항목에 명시했다.

---

## P0 — 배포/통합 전 필수 판단

### P0-1. 멀티태스크(5f) INCART 외부 일반화 역전
- **현상**: P1 확정 멀티태스크 모델(5f)의 INCART(병원 Holter) 이진 응급 AUROC = **0.284**(역전).
  진단상 INCART 실제 AF를 Conduction으로, 정상을 Ischemia로 체계적 오분류 → 응급 평균 0.280 < 정상 0.518.
- **대조**: 동일 INCART·동일 스케일에서 **이진 단독 모델 ③은 역전 없음(AUROC 0.710)**.
  스케일 가설은 기각됨(전역 보정해도 0.281→0.274 불변) — 멀티태스크 5-class 표현의 문제.
- **근거**: `records/03_eval_results.md` §⑪(2026-06-08), 진단 `scripts/_diag_incart.py`, `results/mt_final_snrext.csv`.
- **조치**: 병원 Holter류 도메인 배포 시 **이진 ③ 경로 사용**을 검토. 멀티태스크는 해당 도메인 약점을
  정직히 명시하고 실세계 재검증. P2가 멀티태스크 출력을 쓸 경우 INCART형 분포에서 신뢰도 게이팅 필요.

### P0-2. CACHET 0.99의 낙관성
- **현상**: 웨어러블 AF 목표 도메인 CACHET에서 높은 AUROC(③ 0.844, 5f 0.99대)이나,
  **윈도우 단위·피험자 분리(subject split) 없음** → 절대 성능은 낙관적.
- **근거**: `records/03` §⑦·⑪, `records/05_open_issues.md` 한계 항목.
- **조치**: 웨어러블 절대 성능은 **실세계 종단(피험자 분리) 연구로 재검증 필수**. 상대 비교(모델 간)는 유효.

### P0-3. `embedding[768]` 미노출 — P2 융합 입력 계약 미충족
- **현상**: 명세(`records/00` §5)는 `embedding[768]`을 P2 융합 입력으로 규정하나,
  실제 추론 진입점(`P1CardiacChannel.infer`)·로깅 어댑터(`to_record`) **어디서도 반환하지 않는다**.
  임베딩은 백본 forward에서 이미 `emb = ...mean(1)` (768)로 계산되지만 폐기된다.
- **근거**: `scripts/p1_cardiac_channel.py:124`, `scripts/p1_cardiac_logging_adapter.py`; `MODEL_INTERFACE.md` §4.
- **조치**: P2 연결 전 `infer()`/`to_record()`에 `embedding` 추가 반환(추가 연산 없음, 노출만).
  → P2 작업 차단 요인이므로 P0로 분류.

---

## P1 — 품질·재현성 개선

### P1-1. 의존성 핀 파일 부재
- `requirements.txt`/`environment.yml` 없음. 버전은 문서 기록(PyTorch 2.1.2+cu118, Python 3.9, wfdb 4.3.1 등)에만 존재.
- **근거**: 레포 루트 확인; `README.md`/`records/04`. → **조치**: 검증 환경의 핀 고정 파일 추가.

### P1-2. LoRA 주입 정준 경로 불명확
- 런타임 monkeypatch(`inject_lora`/`_inject`)와 라이브러리 패치(`patches/lora_fairseq_signals.patch`)가 공존.
  현 학습·추론·확정 체크포인트가 쓰는 경로는 런타임 주입으로 보이나 **정준 여부 미검증**.
- **근거**: `ARCHITECTURE.md` §3, `patches/README.md`. → **조치**: 정준 경로 확정·문서화, 불필요 경로 정리.

### P1-3. ECG-FM `.yaml` 사이드카 stale
- 체크포인트 옆 yaml이 차원을 1024/24로 오기(실제 768/12). 도식·자동 설정에 혼동 유발 가능.
- **근거**: `records/ecgfm_backbone_spec.md` 머리말. → **조치**: yaml 무시 주석 또는 정정본 병기.

### P1-4. 이소성(Ectopic) 표현 천장 ~0.82
- ECG-FM frozen mean-pool 표현에서 이소성 AUROC ~0.82 상한. 데이터 보강·pooling 재설계 모두 무효로,
  표현 자체 한계로 결론. 단 **첫-10초 crop 폐기율이 이소성에서 47.6%로 최고** → 윈도우 선택 편향 가능성 잔존.
- **영향**: 이소성=benign(labels_bin=0) → 응급 판정엔 무관(미션 영향 작음).
- **근거**: `records/05`, `records/03` §5e. → **조치(선택)**: 다중윈도우(MIL) 집계 시도(미실행), P2 타이밍·자이로 융합.

### P1-5. mixed_temporal 저SNR 비용
- 배포 권장 레시피(mixed_temporal)는 clean·외부 동등하나 저SNR(6~−6dB)에서 −0.004~−0.010 하락.
  코드 기본값은 재현성 안전을 위해 `single` 유지.
- **근거**: `records/03` §⑩. → **조치**: 실세계 검증에서 재확인, 필요 시 single 폴백.

---

## P2 — P2 연결·후속 작업

### P2-1. physio 계산 명세-구현 드리프트
- `rhythm_regularity`: 명세 `1 - clip(sdNN/200ms)` vs 구현 `1 - clip(cv·3)`(cv=sdNN/mean). 값 정의 불일치.
- `model_version`: 구현 `"lora_multitask_snr_a07"`(접미사 없음) vs 명세 `"_e18"`(§5-1)/`"_e22"`(§5-2, 명세 내부도 비일관).
- 명세가 참조한 `physio_features.py`/`infer_p1.py`는 미존재 — physio는 `p1_cardiac_logging_adapter.py`가 대체 구현.
- **근거**: `MODEL_INTERFACE.md` §4. → **조치**: P2 연결 시 공식 계약(공식·식별자) 1개로 통일.

### P2-2. `benign_flag` 신규 필드 미반영
- `infer`/`to_record`가 반환하는 `benign_flag`(argmax∈{3,4})는 명세 출력 계약에 없음.
- **조치**: P2 소비 규약에 포함 여부 결정·문서화.

### P2-3. `inference_ms` 미구현
- 명세 출력에 포함되나 계측 코드 없음. → **조치**: 필요 시 추론 진입점에 계측 추가.

### P2-4. 2리드(II+V2) 동시추론 미검증
- 로깅 어댑터는 II+V2 raw를 보존하나 **추론은 단일리드 II만**(검증 경로). 2리드 동시추론 임계 미검증.
- **근거**: `p1_cardiac_logging_adapter.py` docstring. → **조치**: Phase-3에서 2리드 실데이터 재검증.

---

## 적용 범위 경계 (모델 한계 아님 — 참고)

> 아래는 "이슈"가 아니라 **설계상 적용 범위 밖**으로 이미 진단·문서화된 항목. 통합 시 오해 방지용.

- **STAFF-III** AUROC ~0.52: balloon inflation 라벨 vs CPSC 진단 기준 미정합 → 비정보적(평가 불가).
- **LTST** AUROC 0.41(③): intra-patient(동일 환자 에피소드 중 vs 외) 태스크 — inter-patient 학습 모델의 범위 외.
- 근거: `records/05_open_issues.md` §1, `records/03` §⑦. 다른 아키텍처(연속 모니터링·개인화) 필요.
