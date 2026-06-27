# HANDOFF_SUMMARY — 인수인계 요약 (한 장)

> 이 레포(`WidU_P1_LoRA-PEFT_Foundation-Model_Adaptation`, ECG-FM+LoRA 응급 심장 이상 탐지)를
> 처음 받는 개발자·통합자를 위한 1페이지 진입 가이드. 상세는 같은 `docs/` 폴더의 4개 문서로 연결.

---

## 1. 이게 뭔가
정적 12-lead로 사전학습된 **ECG-FM(frozen)** 을 **LoRA(q/v, 0.33% 파라미터)** 로만 적응하여,
웨어러블 환경의 **심혈관 응급(AF·급성허혈)** 을 탐지하는 모듈(3-프로젝트 로드맵의 P1).
단일 백본 위에 **이진 응급 헤드 + 5-class 심장 분류 헤드**를 병행하고, 학습 시
**multi-SNR 모션 증강 + RLM lead 마스킹**으로 노이즈·lead 부족에 강건화한다.
출력은 P2(멀티모달 융합)로 전달된다.

## 2. 확정 모델
- **체크포인트**: `outputs/lora_multitask_snr_a07/lora_multitask_snr_best.pt` (α=0.7 BCE, epoch 18).
- **백본**: `checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt` (wav2vec2_cmsc, 90.9M, frozen).
- **핵심 성능**(CPSC mc test): 이진 AUROC **0.9139**, Sens@95%Sp 0.7072, 5-class Macro-F1 **0.6858**.
  단일 이진 참조모델 ③: AUROC 0.9463 / 1-lead 0.9408. (출처 `decisions.md`, `records/03`)

## 3. 핵심 파일 (어디부터 볼까)
| 목적 | 파일 |
|---|---|
| 추론 진입점 | `scripts/p1_cardiac_channel.py` (`P1CardiacChannel.infer`) |
| 로깅 어댑터 | `scripts/p1_cardiac_logging_adapter.py` |
| 학습(확정) | `scripts/train_lora_multitask.py` (`--alpha 0.7`) |
| 증강 | `scripts/multisnr.py` |
| 단일리드 검증 | `scripts/stage0_spine.py` + `tests/test_p1.py` |
| 백본 실측 스펙 | `records/ecgfm_backbone_spec.md` |

## 4. 빠른 실행
```bash
python scripts/verify_env.py                 # 환경 점검
# fairseq-signals: clone @ f8f0ff1 + git apply patches/lora_fairseq_signals.patch + pip install -e .
# 데이터 다운로드/전처리 → 학습 → 평가 (REPRODUCIBILITY.md §5)
```
추론:
```python
from p1_cardiac_channel import P1CardiacChannel
out = P1CardiacChannel().infer(signal_12x5000)   # {emergency_score, cardiac_probs[5], benign_flag}
```

## 5. 입출력 계약 (요점)
- **입력**: `(12,5000)` 또는 `(N,12,5000)` float32, 500Hz·10s, **raw mV(정규화 없음)**,
  N-lead는 0-fill(단일리드는 slot 1=II).
- **출력(실제)**: `emergency_score`(sigmoid) · `cardiac_probs[5]`=[NSR,AF,Ischemia,Conduction,Ectopic](softmax) · `benign_flag`.
- ⚠️ **명세상 `embedding[768]`·`physio`·`model_version`은 명세에 있으나 실제 구현과 드리프트**가 있다.
  특히 `embedding`은 **미노출**(P2 융합 입력 차단) → `MODEL_INTERFACE.md` §4 + `HANDOFF_ISSUES.md` P0-3 필독.

## 6. 통합 전 꼭 알아야 할 것 (TOP 5)
1. **embedding 미노출** — P2 융합 입력 계약 미충족. 노출 추가 필요. (P0-3)
2. **멀티태스크 INCART 역전(0.284)** — 병원 Holter류는 이진 ③ 경로 검토. (P0-1)
3. **CACHET 0.99는 낙관적** — 피험자 분리 없는 윈도우 평가. 실세계 재검증. (P0-2)
4. **LoRA 주입 경로 2가지**(런타임 monkeypatch vs 라이브러리 패치) — 정준 미확정. (P1-2)
5. **`.yaml` 사이드카 stale**(1024/24 ≠ 실제 768/12) / **requirements.txt 없음**. (P1-3, P1-1)

## 7. 문서 지도
- `docs/ARCHITECTURE.md` — 백본·LoRA·헤드·증강·손실·모듈맵·데이터흐름.
- `docs/MODEL_INTERFACE.md` — 추론 입출력 계약(명세 vs 구현 드리프트), P2 소비 규약.
- `docs/REPRODUCIBILITY.md` — 환경·데이터·실행 순서·재현성 갭.
- `docs/HANDOFF_ISSUES.md` — P0/P1/P2 actionable + 적용 범위 경계.
- 연구 상세: `records/00~05`, `decisions.md`, 산출물 `results/`.

## 8. 적용 범위 (정직한 경계)
- **유효**: 12/4/2/1-lead 심혈관 응급(AF·급성허혈) 탐지, 모션 노이즈 강건(−6dB까지).
- **범위 밖**: STAFF-III(라벨 미정합)·LTST(intra-patient 태스크) — 모델 한계 아닌 태스크 구조 불일치.
- **미관측**: 외부충격·저산소는 ECG로 비관측 → P2 융합이 담당(설계 분리).
