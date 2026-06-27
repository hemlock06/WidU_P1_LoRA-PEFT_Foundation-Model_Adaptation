# REPRODUCIBILITY — 환경·데이터·실행 순서

> 환경 구성부터 P1 확정 모델 재현까지의 검증된 절차. 출처: `README.md`,
> `records/04_run_history.md`, `patches/README.md`, `scripts/verify_env.py`,
> `.gitignore`, 각 `download_*`/`preprocess_*`/`train_*` 스크립트.
> 추측 없이 레포에 기록·구현된 사실만 적었고, 누락·갭은 §8에 모았다.

---

## 1. 환경

| 항목 | 값 (출처) |
|---|---|
| OS/GPU | RTX 3060 12GB (`README.md`, `records/04`) |
| Python | 3.9 (`README.md`; 패치 README에 "py39 설치 대응") |
| PyTorch | 2.1.2 + cu118 (`README.md`) |
| 핵심 라이브러리 | `fairseq_signals`(editable), `torch`, `numpy`, `scipy`, `scikit-learn`, `wfdb`(4.3.1), `h5py` |
| RAM 제약 | 16GB 기준으로 mmap·순차 로드 설계(`records/04`, `train_lora_multitask.py`) |

> ⚠️ **`requirements.txt`·`environment.yml` 없음** (레포 확인). 버전은 위 문서 기록에 의존한다.
> 핀 고정 의존성 파일 생성이 권장됨 → `HANDOFF_ISSUES.md` P1.

환경 점검: `python scripts/verify_env.py` → `fairseq_signals.__version__`, `torch.__version__`,
`torch.cuda.is_available()`, GPU 이름을 출력.

---

## 2. fairseq-signals (LoRA 패치) 설치

ECG-FM 백본은 `fairseq-signals` 라이브러리로 로드한다. 이 라이브러리는 용량 문제로
git 미추적(`.gitignore: fairseq-signals/`)이며, 본 프로젝트의 LoRA 수정분은
`patches/lora_fairseq_signals.patch`로 보존된다.

```bash
# 1. base commit으로 clone (patches/BASE_COMMIT.txt)
git clone https://github.com/Jwoo5/fairseq-signals.git
cd fairseq-signals
git checkout f8f0ff1c788a82c2059cb452cd5462898867489e
# 2. 패치 적용
git apply ../patches/lora_fairseq_signals.patch
# 3. py39 env에 editable 설치
pip install --editable ./
```

패치 포함 내용: `multi_head_attention.py`(LoRA 주입), `ecg_transformer_classifier.py`(LoRA config),
`setup.py`(py39 대응). (출처: `patches/README.md`)

> 주의: 학습·추론 스크립트는 LoRA를 **런타임 monkeypatch**로도 주입한다(`ARCHITECTURE.md` §3).
> 두 경로의 관계는 미검증이므로, 라이브러리 패치 없이 런타임 주입만으로 동작하는지는 통합 시 확인 필요.

---

## 3. ECG-FM 체크포인트

- 출처: HuggingFace `bowang-lab/ecg-fm` (`records/04` 환경 설정).
- 배치: `checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt` (백본으로 사용).
- `checkpoints/`는 git 미추적(`.gitignore`) — 별도 다운로드 필요.
- ⚠️ 동봉 `.yaml` 사이드카는 차원이 stale(1024/24); 실제는 768/12 (`ARCHITECTURE.md` §2).

---

## 4. 데이터셋

모든 전처리 출력 공통 규격: float32 `(N, 12, 5000)` @500Hz, 라벨 int8 `(N,)`,
split seed=42 (70/15/15). `data/`·`outputs/`는 git 미추적(대용량).

| 데이터셋 | 다운로드 | 전처리 | 출력 | split | 용도 |
|---|---|---|---|---|---|
| **CPSC 2018** | `download_cpsc2018.py` | `preprocess_cpsc2018.py`(이진) / `preprocess_cpsc2018_mc.py`(5-class) | `data/processed/cpsc2018/`, `.../cpsc2018_mc(_ml)/` | record-level (patient ID 미제공) | ★ 학습·테스트 주 데이터 |
| **PTB-XL** | `download_ptbxl.py` | `preprocess_ptbxl.py` | `data/processed/ptbxl/` | patient-level | 이진 외부 척도 확장(5c) |
| **SPH** | (figshare tar.gz) | `preprocess_sph.py` | `data/processed/sph/` | patient-level | 급성허혈 보강(5-class) |
| **NSTDB** | `download_nstdb.py` | (런타임 `MultiSNRNoise` 직접 로드) | `data/raw/nstdb/{bw,em,ma}` | — | multi-SNR 증강 노이즈 |
| **CACHET-CADB** | `download_cachet.py` | `preprocess_cachet.py` | `data/processed/cachet/` | 없음(inference-only) | 외부검증(웨어러블 AF, 1-lead) |
| **INCART** | `download_external_dbs.py` | `preprocess_incart.py` | `data/processed/incart/` | 없음 | 외부검증(병원 Holter 12-lead) |
| **STAFF-III** | `download_external_dbs.py` | `preprocess_staffiii.py` | `data/processed/staffiii/` | 없음 | 외부검증(장기 ST, 9-lead) |
| **LTST** | `download_external_dbs.py` | `preprocess_ltst.py` | `data/processed/ltst/` | 없음 | 외부검증(허혈 에피소드, 2-lead) |

**전처리 핵심 파라미터** (출처: 각 스크립트 / `records/04`):
- 윈도우: 앞 10초(5000샘플) crop, 부족분 우측 zero-pad. 리샘플 → 500Hz.
- 정규화: **미적용**(ECG-FM 사전학습 정합). NaN/Inf: CPSC·PTB-XL는 레코드 스킵, 장기 레코드는 `nan_to_num`.
- CPSC 이진 라벨(옵션 A): 응급(1)=AF·STD·STE / 정상(0)=Normal / 제외=I-AVB·LBBB·RBBB·PAC·PVC.
- CPSC 5-class: `[NSR, AF, STD/STE, I-AVB/LBBB/RBBB, PAC/PVC]`.

**검증된 split 수** (`records/04`):
- CPSC 이진: train 2217 / val 474 / test 474. CPSC 5-class(`cpsc2018_mc`): train 4357 / val 933 / test 936.
- PTB-XL: train 12847 / val 2790 / test 2742.
- 외부검증 윈도우 수: CACHET (1362, 12, 5000) / INCART (7811,...) / STAFF-III (6650,...) / LTST 15426 윈도우.

> ⚠️ `cpsc2018_mc`(단일라벨, 5b/5d/5f 학습) vs `cpsc2018_mc_ml`(multi-label, 5e) 구분 주의.
> 5e는 record 복구로 구 split과 어긋나 **데이터 누수**가 확인됨(`records/03` §5e) — 공정 비교는 clean subset 사용.
> **P1 확정(5f)은 `cpsc2018_mc`로 학습**되었다(`train_lora_multitask.py` `DATA_DIR`).

---

## 5. 실행 순서 (P1 확정 모델 재현)

```bash
# 0) 환경 점검
python scripts/verify_env.py

# 1) 데이터 다운로드 (시간 소요: CPSC ~83분 등, records/04)
python scripts/download_cpsc2018.py --workers 8
python scripts/download_nstdb.py
python scripts/download_ptbxl.py --workers 8          # (혼합 학습/보조용)
python scripts/download_cachet.py
python scripts/download_external_dbs.py               # INCART / STAFF-III / LTST

# 2) 전처리
python scripts/preprocess_cpsc2018_mc.py              # → data/processed/cpsc2018_mc/
python scripts/preprocess_cachet.py
python scripts/preprocess_incart.py
python scripts/preprocess_staffiii.py
python scripts/preprocess_ltst.py
# (선택) preprocess_ptbxl.py, preprocess_sph.py, build_cpsc_sph.py

# 3) P1 확정 모델 학습 (α=0.7, multi-SNR)
#    warm start = outputs/lora_multitask/lora_multitask_best.pt (5d, 사전 학습 필요)
python scripts/train_lora_multitask.py \
    --alpha 0.7 \
    --out_dir outputs/lora_multitask_snr_a07
#    → outputs/lora_multitask_snr_a07/lora_multitask_snr_best.pt (best composite, epoch 18)
#    재현 옵션: --noise_mode {single|mixed|mixed_temporal} (배포 권장 mixed_temporal)
#              --resume (OOM 시 self-healing 재개, rng 복원 → 결정적)

# 4) 평가
python scripts/eval_multitask_test.py        # CPSC mc test 지표
python scripts/eval_snr_curve.py             # SNR 저하 곡선
python scripts/eval_external.py              # 외부검증 4종
python scripts/ablation_nlead_curve.py       # N-lead 강건성
```

> 참고: `--alpha`/`--out_dir`는 `train_lora_multitask.py`의 실제 argparse 인자다.
> warm start 기본값(`WARM_CKPT = outputs/lora_multitask/lora_multitask_best.pt`)이 필요하므로
> 5d 멀티헤드 체크포인트를 먼저 만들거나(`--warm_ckpt ""`로 cold start) 기존 체크포인트를 배치해야 한다.
> 정확한 인자·기본값은 스크립트 `main()`의 argparse 정의를 신뢰할 것.

---

## 6. records/ 요약 (상세 기록 위치)

| 파일 | 내용 |
|---|---|
| `records/00_research_plan.md` | 3-프로젝트 로드맵, P1 다중출력 설계, **§5 P1→P2 인터페이스 명세** |
| `records/01_design_decisions.md` | 아키텍처·LoRA·RLM·멀티태스크 Path B·α=0.7 근거, 전처리 7원칙 |
| `records/02_training_logs.md` | 전 에폭 훈련 로그(①②③ 이진, 5b/5d/5e/5f) |
| `records/03_eval_results.md` | 테스트 지표·SNR 곡선·외부검증 4종·5b/5d/5e/5f 비교·단일리드·INCART 역전 규명(⑪) |
| `records/04_run_history.md` | 환경·다운로드·전처리 이력(LTST OOM 수정 v1~v3 포함) |
| `records/05_open_issues.md` | 범위 경계·한계·이슈 트래커 |
| `records/ecgfm_backbone_spec.md` | ECG-FM 백본 구조 실측(도식 검증), yaml stale 경고 |

산출물 CSV·도표: `results/` (snr_curve, nlead_curve, external_eval_*, mt_*_metrics 등),
체크포인트 위치 목록은 `decisions.md` "체크포인트 위치".

---

## 7. 재현성 주의 — 비결정성

- cuDNN 비결정 연산으로 실행 간 **AUROC ±~0.002 변동**(`records/05`). 4자리 정밀 재현 불가.
  결론은 변동폭보다 훨씬 큰 차이(+0.02~+0.04)에 근거하므로 신뢰 가능.
- 학습 self-healing resume는 rng 상태(np/torch/cuda/multisnr)까지 저장·복원하여
  무중단과 동일한 realization을 보장(`train_lora_multitask.py`).

---

## 8. 재현성 갭 (인수인계 확인 항목)

1. **의존성 핀 파일 부재** — `requirements.txt`/`environment.yml` 없음(§1). → `HANDOFF_ISSUES.md` P1.
2. **fairseq-signals 외부 의존** — clone+patch+editable 설치 필요(§2), 런타임 주입과의 관계 미검증.
3. **데이터·체크포인트 미추적** — `data/`·`outputs/`·`checkpoints/`·`*.pt` 모두 .gitignore. 재다운로드/재학습 필요.
4. **warm start 체인** — 5f는 5d 체크포인트를 warm start로 가정(§5). 전체 cold 재현 경로는 명시 필요.
5. **SPH 다운로드 스크립트** — `preprocess_sph.py`는 tar.gz HDF5를 직접 소비; 명시적 `download_sph.py`는 미확인.
