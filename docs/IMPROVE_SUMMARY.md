# IMPROVE_SUMMARY — P0/P1 코드 개선 결과

> 일시: 2026-06-28 / 브랜치: `improve` (← `handoff-prep` 분기, main·handoff-prep 미변경).
> 범위: `docs/HANDOFF_ISSUES.md` 의 P0/P1 중 **안전·검증가능 + 이슈에 명시된** 항목만 구현.
> 설계결정·미검증 항목은 `docs/IMPROVE_PROPOSALS.md` 에 옵션으로 분리.
> 환경 제약: 본 작업 환경은 **torch 미설치**(Python 3.11, hermes venv) — 모델 추론 실행 불가.
> 따라서 추론을 직접 실행해야 하는 검증은 불가했고, 그 사실을 항목별로 명시했다.

---

## 1. 한 일 (적용된 변경)

### ① P0-3 — `infer()` 에 embedding[768] 노출 *(코드 변경)*
- **파일**: `scripts/p1_cardiac_channel.py` (`P1CardiacChannel.infer`).
- **내용**: 추론 루프에서 두 헤드 입력으로 **이미 계산되어 폐기되던** mean-pool 임베딩
  `emb (b,768)` 을 배치별 누적(`zb`)하여 반환 dict 에 `"embedding"` 키로 추가.
  단일 입력 → `(768,)`, 배치 → `(N,768)`. **추가 forward·연산 없음(노출만)**.
- **순수 가산적**: 기존 키(`emergency_score`/`cardiac_probs`/`benign_flag`) 계산·타입·순서 불변.
- **근거**: HANDOFF_ISSUES P0-3, `ARCHITECTURE.md` §7(미노출 명시), `records/00` §5(P2 융합 입력 규정).

### ② P1-1 — 의존성 핀 파일 추가 *(재현성)*
- **파일**: `requirements.txt`, `environment.yml` (신규, 레포 루트).
- **내용**: **문서로 검증된 버전만** 정확히 핀 — `torch==2.1.2+cu118`, `wfdb==4.3.1`, `python=3.9`.
  버전이 어디에도 기록되지 않은 패키지(numpy/scipy/scikit-learn/h5py/pandas/requests/tqdm/
  matplotlib/pyarrow)는 **추측 금지** 원칙상 이름만 명시(미핀)하고 그 사실을 주석으로 라벨.
  `fairseq_signals`(PyPI 미배포)는 clone+패치+editable 설치 절차를 주석으로 기재.
- **근거**: HANDOFF_ISSUES P1-1, `REPRODUCIBILITY.md` §1·§8-1, `README.md` L107, 각 스크립트 import.

### ③ 회귀/검증 테스트 추가
- `tests/test_requirements_pins.py` — 핀 파일 형식(PEP 508 파싱)·검증된 버전 고정·python=3.9 단정.
  **torch 불필요 → 이 환경에서 실제 실행됨**.
- `tests/test_embedding_contract.py` — `infer()` 의 embedding[768] 출력 계약 단정.
  torch+체크포인트 필요 → 없으면 자동 skip(검증 환경용 회귀 가드).

---

## 2. 테스트 결과

| 테스트 | 결과 | 비고 |
|---|---|---|
| `tests/test_requirements_pins.py` (4 케이스) | **4 passed** | torch 없이 실행되는 독립검증 |
| `tests/test_embedding_contract.py` (2 케이스) | **skipped** | torch 미설치 → 검증 환경에서 실행 필요 |
| `py_compile` (변경/신규 .py 전체) | **OK** | 구문·import 컴파일 통과 |
| 적대적 diff 리뷰 (P0-3, 새 컨텍스트 서브에이전트) | **infer 변경 머지가능** | 항목 1~6 OK. 7번(to_record 직렬화) 지적 → 아래 반영 |

> ⚠️ **기존 테스트 스위트(`tests/test_p1.py`)는 이 환경에서 수집 실패**한다 —
> `stage0_spine.py` 가 모듈 최상단에서 `import torch` 하는데 torch 미설치이기 때문(사전 조건,
> 본 변경과 무관). 이로 인해 `pytest tests/` 전체 실행은 collection 에서 중단되므로,
> 신규 테스트는 파일 지정(`pytest tests/test_requirements_pins.py`)으로 실행·검증했다.

---

## 3. 미검증 (적용했으나 이 환경에서 런타임 실행으로 확인 못 한 것)

- **P0-3 `infer()` embedding 노출의 런타임 동작**: torch·ECG-FM 체크포인트 부재로 실제 추론을
  돌려 shape/값을 확인하지 못함. 대신 **(a) py_compile 통과 (b) 기존 es/cp 누적 패턴과 동일 구조
  (c) 새 컨텍스트 적대적 정적 리뷰(머지가능 판정)** 로 독립검증. 검증 환경에서
  `tests/test_embedding_contract.py` 를 실행하면 계약이 자동 단정된다.
- **의존성 핀의 실제 설치 가능성**: `torch==2.1.2+cu118` 등의 휠 설치는 네트워크·GPU 환경이
  필요해 실행하지 않음. 검증한 것은 **형식 유효성 + 문서 기록과의 일치**뿐.

---

## 4. 미구현 (제안으로만 — `docs/IMPROVE_PROPOSALS.md`)

설계결정이 필요하거나 이 환경에서 독립검증이 불가하여 **구현하지 않고 제안**으로 남긴 항목:

- **A. `to_record()` embedding 영속화** — 스칼라-only 레코드에 (768,) 배열 주입은 직렬화/parquet
  스키마 설계결정. 적대적 리뷰가 직렬화 리스크를 지적했고 이 환경에서 parquet 경로 실행검증
  불가 → **되돌리고** 옵션 3종(인라인/별도store/플래그)으로 제안. (`infer()` 노출은 유지)
- **B. P0-1** INCART 역전 → 배포 라우팅 정책(이진 ③ 경로) 설계결정.
- **C. P0-2** CACHET 절대 성능 → 실세계 피험자분리 재검증(코드 아님).
- **D. P1-2** LoRA 정준 경로 확정 → 검증 환경 실행 필요(torch 부재로 불가).
- **E. P1-4** 이소성 천장 → 선택적(MIL/융합), 미션 영향 작음.
- **F. P1-5** mixed_temporal 저SNR → 검증 후 기본값 결정.
- **G. P2-1~4** P2 연결 후속(범위 밖, 인용만).

### 충족됐으나 추가 변경 불필요
- **P1-3 stale yaml(1024/24 vs 실제 768/12)**: 경고가 이미 `ARCHITECTURE.md` §2,
  `REPRODUCIBILITY.md` §3, `records/ecgfm_backbone_spec.md` 머리말에 명시돼 있다.
  yaml 사이드카 자체는 `.gitignore` 로 **비추적**(checkpoints/ 전체 제외)이라 그 위치에 커밋 가능한
  변경을 둘 수 없다 → 추적 문서 경고로 이미 충족, 추가 커밋 변경 없음.

---

## 5. 안전장치 준수

- `main`·`handoff-prep` **미변경**(분기만). 모든 변경은 `improve` 브랜치. push 안 함.
- 인증/인가·상태영속성·DB·푸시전달 등 **설계결정 항목은 구현하지 않고 제안만**.
- 독립검증 불가 변경(`to_record` 영속화)은 **잔류시키지 않고 되돌림**(미검증 코드 잔류 금지).
