# 실행 기록 (Run History)

> Pre-flight, 데이터 다운로드, 환경 설정 등 학습 외 실행 기록

---

## Pre-flight 1: PhysioNet 2011 per-channel 라벨 확인 (2026-05-25)

**목적**: 게이트(단계 7) 학습에 필요한 per-lead(채널별) 라벨이 공식 release에 있는지 확인

**실행**: `scripts/preflight_1_physionet2011.py --data_dir D:/WidU_ecg-fm_emergency-detection/data/raw/physionet2011`

### 데이터 구조

| 항목 | set-a | set-b |
|---|---|---|
| 총 파일 | 3,004개 | 1,502개 |
| .dat | 1,000개 | 500개 |
| .hea | 1,000개 | 500개 |
| .txt | 1,000개 | 500개 |

| 파일 | 레코드 수 |
|---|---|
| RECORDS | 1,000개 |
| RECORDS-acceptable | 773개 |
| RECORDS-unacceptable | 225개 |

### 조사 결과

| 확인 항목 | 결과 |
|---|---|
| .txt 파일 내 per-channel grade (A/B/C/D/F) | **미발견** — .txt는 raw signal (13컬럼 = time + 12 leads) |
| .hea 파일 내 quality 주석 | **없음** — `#<age>: 0 <sex>: ?` 메타만 존재 |
| per-channel grade 키워드 | **미발견** |

### 결론 및 후속 전략

- **per-channel grade 없음**: PhysioNet 2011 공식 release에는 record-level acceptable/unacceptable만 제공
- **단계 7 전략**: PhysioNet 2011 record-level label (acceptable=양호, unacceptable=불량)을 게이트 학습 라벨로 직접 사용
  - acceptable 773개 → 품질 양호(1)
  - unacceptable 225개 → 품질 불량(0)
  - 형식: 12-lead, 500Hz, 5000샘플 — ECG-FM 입력과 완벽 일치

---

## Pre-flight 2: ECG-FM 4-lead 0-fill forward 테스트 (2026-05-24)

**목적**: 12-lead 슬롯에 N개 lead만 넣고 나머지 0-fill → forward 정상 작동 여부 확인

**실행**: `scripts/preflight_2d_features_only.py`

**결과**: PASS — 모든 lead 구성에서 NaN/Inf 없음. API 확정. (상세: `records/01_design_decisions.md` Pre-flight 2 섹션 참조)

---

## CPSC 2018 데이터 다운로드 (2026-05-24)

**실행**: `python scripts/download_cpsc2018.py --workers 8` (~83분 소요)

| 항목 | 값 |
|---|---|
| 총 대상 파일 | 13,755개 |
| 다운로드 성공 | 13,704개 |
| 오류 (타임아웃) | 51개 |
| 저장 .hea | 6,839개 |
| 저장 .mat | 6,865개 |

- REFERENCE.csv 타임아웃 실패 → `.hea` `#Dx:` 파싱으로 대체
- 오류 51건: 모두 `WinError 10060` (네트워크 타임아웃)
- 결손 .mat: A5883, A4795, A6020, A1251, A1524 (전처리 시 스킵)

---

## NSTDB 다운로드 (2026-05-24)

**목적**: multi-SNR 증강용 노이즈 파형 (PhysioNet MIT-BIH Noise Stress Test Database)

**결과**: bw / em / ma 3종 다운로드 완료
- 원본 360Hz → 500Hz 리샘플 (wfdb.processing.resample_sig)
- pool 길이: 1,805,556 samples per type

---

## PhysioNet 2011 다운로드 (2026-05-25)

**목적**: 단계 7 신호품질 게이트 학습 데이터

**실행**: `python scripts/download_physionet2011.py`

| 항목 | 값 |
|---|---|
| set-a.tar.gz | ~103MB, 3,004개 파일 |
| set-b.tar.gz | ~51MB, 1,502개 파일 |
| 저장 경로 | `D:/WidU_ecg-fm_emergency-detection/data/raw/physionet2011` |

---

## CACHET-CADB 다운로드 (2026-05-25)

**목적**: 단계 9 외부검증 held-out (AF/NSR 1,362개 유효)

**실행**: `python scripts/download_cachet.py`
**출처**: DTU Data (DOI: 10.11583/DTU.14547330.v1), figshare URL

| 항목 | 값 |
|---|---|
| 파일명 | `cachet-cadb_short_format_without_context.hdf5` |
| 용량 | 262,473,728 bytes (262.5MB) |
| HDF5 구조 | `signal` (16,404,480,) + `labels` (16,404,480,) |
| 샘플 수 | 1,602개 (10s × 1024Hz = 10,240 pts) |
| 라벨 분포 | AF=747, NSR=615, Noise=221, Others=19 |
| Patient ID | **없음** (Short Format은 flat 구조) |

---

## 외부검증 DB 다운로드 (2026-05-25)

**목적**: 단계 9 외부검증 (INCART, STAFF-III, LTST)

**실행**: `python scripts/download_external_dbs.py`

| DB | 용량 | 상태 | 저장 경로 |
|---|---|---|---|
| INCART | 563MB | 완료 | `data/raw/incart` (75 레코드) |
| STAFF-III | 3.2GB | 완료 | `data/raw/staffiii` (516 .hea) |
| LTST | 6.73GB | 완료 (86/86 레코드, 2026-05-26) | `data/raw/ltst` |

---

## 단계 9 외부검증 전처리 (2026-05-26)

**목적**: 다운로드 완료된 외부검증 DB 3종을 단계 9 추론용 `.npy`로 전처리 (12-lead, 500Hz, 5,000샘플 윈도우)

**환경**: `D:/conda_envs/py39/python.exe` (wfdb 4.3.1, scipy, h5py)

| DB | 스크립트 | 출력 shape | 응급 | 정상 | 비고 |
|---|---|---|---|---|---|
| CACHET-CADB | `preprocess_cachet.py` | (1362, 12, 5000) | 747 (AF) | 615 (NSR) | 1,602세그 중 Noise/Others 240 제외; 단일채널→lead II 슬롯 0-fill |
| INCART | `preprocess_incart.py` | (7811, 12, 5000) | 540 (AFIB) | 7271 | AFIB 레코드 I49/I50/I71(WPWAF) 전 윈도우 응급; 정상 후보 69개 N≥90% 필터 |
| STAFF-III | `preprocess_staffiii.py` | (6650, 12, 5000) | 3550 ('b') | 3100 ('a') | 516 .hea 중 a=104/b=102 선택, 나머지 c~f 310 제외; 9-lead→12-lead 재배열(aVR/aVL/aVF 0-fill) |

**저장 경로**: `data/processed/{cachet,incart,staffiii}/{signals.npy, labels.npy[, record_ids.npy]}`

---

## CPSC 2018 다중분류 전처리 (2026-05-26)

**목적**: 단계 5b 심장 다중분류 학습용 — 기존 이진(응급/정상/제외) 대신 5-class 레이블로 재전처리

**실행**: `python scripts/preprocess_cpsc2018_mc.py`

| 항목 | 값 |
|---|---|
| 스크립트 | `scripts/preprocess_cpsc2018_mc.py` |
| 입력 | `data/raw/cpsc2018` (동일 원본) |
| 출력 | `data/processed/cpsc2018_mc/` |
| Split | train=4357 / val=933 / test=936 |
| 변경점 | 기존 제외(-1)였던 전도장애·이소성 클래스를 고유 클래스(3·4)로 복귀 |

**클래스 분포 (train)**

| 클래스 | n |
|---|---|
| 정상(NSR) | 640 |
| AF | 820 |
| 허혈성(STD/STE) | 755 |
| 전도장애(I-AVB/LBBB/RBBB) | 1770 |
| 이소성(PAC/PVC) | 372 |

---

## LTST 전처리 v1~v3 — OOM 수정 이력 (2026-05-26)

**목적**: LTST-DB(Long-Term ST) → 단계 9 추론용 `.npy` 변환

### v1 (실패 — FFT OOM)

**실행**: `D:\conda_envs\py39\python.exe scripts/preprocess_ltst.py` (PID 690)

- `scipy.signal.resample` (FFT 기반) 사용
- s20011~s20071 정상 처리 후 **s20081부터 연속 `std::bad_alloc`**
- 원인: 45시간 레코드 FFT 중간 복소수 배열 (~309 MiB complex64) 할당 실패
- 처리 결과: 응급 2,058 / 정상 363 (8레코드 성공, ~10레코드 실패)

### v2 (부분 실패 — 12-lead 배열 OOM)

**패치**: `sp_resample` → `resample_poly` (polyphase 필터, FFT 대비 메모리 효율적)

**실행**: PID 702

- 리샘플 단계 OOM 해소 → s20081~s20111 통과 - **s20121부터 재실패**: `np.zeros((12, T_out), float32)` 단계에서 OOM
  - 원인: 45시간 레코드 T_out=81,000,000 → (12, 81M) × 4B = **3.7 GB 단일 배열**
- 12-lead 전체 배열을 한번에 만드는 구조 자체가 문제

### v3 (성공 — 윈도우 단위 배치)

**패치**: `process_record` 반환값 변경 + `make_window_12lead` 추가

- `process_record` → `(sig_rs, slot_indices, n_windows)` 반환 (12-lead 배열 제거)
- 윈도우 루프 내 `make_window_12lead(sig_rs, slot_indices, w)` → (12, 5000) = 240 KB/윈도우
- `del sig_rs`: 레코드 처리 후 즉시 해제

**결과**: 86/86 전체 통과 | 항목 | 값 |
|---|---|
| 유효 윈도우 | 15,426개 |
| 응급(허혈 ST) | 11,346 |
| 정상 | 4,080 |
| 처리 레코드 | 86/86 |
| 출력 | `data/processed/ltst/signals.npy` (3.53 GB), `labels.npy`, `record_ids.npy` |

---

## PTB-XL 다운로드 + 혼합 학습 시행착오 (2026-05-27)

### PTB-XL 다운로드

- 스크립트: `scripts/download_ptbxl.py` (멀티스레드 8워커, 재시도 로직)
- 1차 실행: 42,862개 신규 다운로드, 5개 네트워크 타임아웃 실패
- 2차 재시도: 5개 복구, 최종 21,798 .dat / 21,837 목표 (99.8%)
- 나머지 39개: records500에 500Hz 버전 없는 레코드 — 전처리에서 자동 스킵

### PTB-XL 전처리

- 스크립트: `scripts/preprocess_ptbxl.py`
- `conda run` 사용 시 UnicodeEncodeError (한글 stdout + cp949 충돌) → `D:/conda_envs/py39/python.exe` 직접 실행으로 우회
- 결과: train=12,847 / val=2,790 / test=2,742 (patient-level split, seed=42)
- 클래스 분포: 응급(MI/STTC)=9,285 / 정상(NORM)=9,097 / 제외=3,417

### 혼합 학습 시행착오 1 — RAM 포화 (2026-05-27 01:18~)

**증상**: 학습 시작 106분 경과, 출력 파일 0 bytes, GPU 100%, RAM 잔여 0.8GB

**원인**:
- 외부검증 4종(CACHET/INCART/STAFF-III/LTST)을 `ECGDataset.__init__`에서 전부 RAM에 선로드
- LTST 3.5GB + INCART 1.9GB + STAFF-III 1.6GB + PTB-XL train 3.1GB + 모델 1.7GB ≈ 14GB → RAM 포화 → OS 스왑
- stdout block buffering (파일 리다이렉트 시 Python이 8KB 단위 버퍼링) → 출력 0 bytes처럼 보임

**조치** (`scripts/train_lora_mixed.py` 수정):
- 에폭별 외부검증 제거 → **최종 1회, 순차 로드 후 즉시 해제** (`del ext_ds, ext_loader`)
- `-u` 플래그 추가 (unbuffered stdout)
- 헤더 단순화 (외부검증 열 제거)

### 혼합 학습 시행착오 2 — Cold start AUROC ~0.5 (2026-05-27 01:09~)

**증상**: 에폭 1~5 val AUROC 0.48~0.53 (랜덤 수준), CPSC 단일 학습은 에폭 1부터 0.85+

```
Ep  Loss    ValAUROC  ValF1
 1  1.0119  0.4833    0.000
 2  0.6786  0.5252    0.831
 3  0.6781  0.5035    0.000
 4  0.6747  0.4952    0.000
 5  0.6698  0.4775    0.831
```

**원인**:
- LoRA 랜덤 초기화 + PTB-XL(12,847) : CPSC(2,217) = 6:1 비율
- 그래디언트가 PTB-XL 특성 방향으로 지배 → CPSC val에서 랜덤 수준 성능
- LR=5e-4가 cold start에는 적절하지만 CPSC 지식 유지에 불리

**조치**:
- **Warm start**: `lora_multisnr_best.pt` (CPSC 단일, val AUROC=0.9525) 로드 후 파인튜닝
- LR=1e-4 (기본값 낮춤, CPSC 지식 보존)
- `--ckpt_start` 인자 추가 (경로 지정 가능)

### 혼합 학습 시행착오 3 — Warm start 후 catastrophic forgetting (2026-05-27)

**증상**: Warm start 후 에폭 1은 0.8131로 정상, 에폭 2~3에서 0.42→0.36으로 폭락

```
Ep  Loss    ValAUROC
 1  0.5637  0.8131 ←
 2  0.7515  0.4213
 3  0.6979  0.3566
```

**원인**:
- LR=1e-4가 PTB-XL 그래디언트 규모 대비 여전히 높음
- ptbxl_ratio=1.0 → 에폭당 CPSC:PTB-XL = 1:1 샘플링이지만
  PTB-XL 특성(MI/STTC)이 CPSC 특성(AF/STD)과 달라 한 에폭 만에 CPSC 지식 덮어씀
- Catastrophic forgetting 전형 패턴

**조치**:
- LR=1e-5 (10배 낮춤)
- ptbxl_ratio=0.3 (PTB-XL 샘플 CPSC 대비 30%로 축소, CPSC 그래디언트 지배)
- `python train_lora_mixed.py --lr 1e-5 --ptbxl_ratio 0.3`

### 혼합 학습 성공 — 4차 (2026-05-27)

**설정**: `--lr 1e-5 --ptbxl_ratio 0.3 --ckpt_start lora_multisnr/lora_multisnr_best.pt`

**결과**: 30 에폭 완주, best epoch 7 val AUROC=0.9748

| Ep | ValAUROC | F1 |
|----|----------|------|
| 1 | 0.9664 | 0.9285 |
| 2 | 0.9714 | 0.9391 |
| **7** | **0.9748** ← best | 0.9445 |
| 30 | 0.9674 | 0.9085 |

**최종 평가 (best ckpt 기준, 자세한 비교는 records/03 단계 5c 참조)**:
- CPSC test AUROC=0.9714 (단일 ③ 0.9463 대비 +0.025)
- 외부 4종 (CACHET/INCART/STAFF-III/LTST): 단일 대비 ±0.02 이내 (큰 차이 없음)

**4차에서 학습이 안정된 원인**:
- LR을 5e-4 → 1e-5로 50배 낮춤 → PTB-XL 그래디언트 규모 통제
- ptbxl_ratio 1.0 → 0.3 → CPSC 신호 우세 유지
- warm start: lora_multisnr의 CPSC 학습된 표현 보존

**연구방향 정합성 (사후 검토)**:
- 본 모델은 **이진** 출력. 단계 5b 다중분류와 별도 백본.
- PTB-XL의 SCP 코드는 5b의 5-class taxonomy(NSR/AF/허혈/전도/이소성)로 매핑 가능 → 향후 **다중분류 혼합 학습**이 본래 연구 의도(`records/00_research_plan.md` §1: "동일 백본 위에 두 헤드 병행")에 부합.
- 현 혼합 이진 모델은 외부 평가 척도 확장 자료로 보존.

---

## 환경 설정 기록 (2026-05-24)

- fairseq-signals: `pip install -e .` (editable install, C:\ecg-project\fairseq-signals)
- ECG-FM 체크포인트: HuggingFace `bowang-lab/ecg-fm` → `mimic_iv_ecg_physionet_pretrained.pt`
- h5py 설치: `pip install h5py` (CACHET HDF5 탐색용)
- wfdb 설치: `pip install wfdb` (PhysioNet 다운로드·로드용)
