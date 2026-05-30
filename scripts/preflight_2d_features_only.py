"""
Pre-flight 2d: ECG-FM features_only=True 로 올바른 768-dim 임베딩 추출 + 0-fill 검증
===================================================================================
이전 결과 분석:
  - 모델 로드: Wav2Vec2CMSCModel 90.9M
  - Inf 발생 원인: features_only=False(기본값) → quantizer가 더미 노이즈에 수치 폭발
  - shape (101,1,149): contrastive head 출력 — fine-tuning에 쓰는 경로 아님

수정 사항:
  - model(source=x, padding_mask=None, features_only=True) 사용
  - 예상 출력: {'x': (T, B, 768), 'padding_mask': ...}
  - 이 'x'가 fine-tuning / 분류 헤드 입력이 되는 768-dim 임베딩

사용법:
  python scripts/preflight_2d_features_only.py
    --checkpoint checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt
"""

import argparse
import os
import traceback

import numpy as np
import torch

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 500
WINDOW_LEN = FS * 10  # 5000 샘플
EMBED_DIM = 768        # ECG-FM transformer 출력 차원


def make_dummy_ecg(seed=42):
    """더미 ECG 생성. 0-평균, 0.3 std (mV 단위 근사)."""
    rng = np.random.RandomState(seed)
    return rng.randn(12, WINDOW_LEN).astype(np.float32) * 0.3


def apply_zero_mask(signal, available_leads):
    masked = signal.copy()
    for i in range(12):
        if i not in available_leads:
            masked[i, :] = 0.0
    return masked


def load_model(ckpt_path, device):
    from fairseq_signals.utils.checkpoint_utils import load_model_and_task
    # 진단에서 확인된 올바른 시그니처: load_model_and_task(filename, ...)
    result = load_model_and_task(ckpt_path)
    # result는 tuple — 첫 번째 모델 객체 추출
    if isinstance(result, tuple):
        for r in result:
            if hasattr(r, "parameters"):
                return r.to(device)
            if isinstance(r, list) and r and hasattr(r[0], "parameters"):
                return r[0].to(device)
    return result.to(device)


def run_forward(model, x):
    """
    features_only=True 로 768-dim 인코더 임베딩만 추출.
    여러 호출 패턴을 시도해서 성공한 결과 반환.
    """
    call_patterns = [
        # 패턴 1: features_only 키워드 (fairseq-signals 표준)
        lambda m, t: m(source=t, padding_mask=None, features_only=True),
        # 패턴 2: extract_features 메서드 (fairseq 일부 버전)
        lambda m, t: m.extract_features(source=t, padding_mask=None),
        # 패턴 3: features_only=True, mask=False 추가
        lambda m, t: m(source=t, padding_mask=None, features_only=True, mask=False),
        # 패턴 4: mask=False 만
        lambda m, t: m(source=t, padding_mask=None, mask=False),
        # 패턴 5: 그냥 forward (마지막 수단, Inf 가능성 있음)
        lambda m, t: m(source=t, padding_mask=None),
    ]

    for i, pattern in enumerate(call_patterns, 1):
        try:
            with torch.no_grad():
                out = pattern(model, x)

            # 출력 딕셔너리에서 'x' 키 우선 추출
            if isinstance(out, dict):
                # 모든 키와 shape 출력 (첫 번째 성공 케이스에서만)
                if i <= 3:
                    print(f"    출력 dict 키/shape:")
                    for k, v in out.items():
                        if isinstance(v, torch.Tensor):
                            print(f"      '{k}': {tuple(v.shape)}")
                        elif v is None:
                            print(f"      '{k}': None")

                # 'x' 키: 인코더 출력 (T, B, 768)
                if "x" in out and isinstance(out["x"], torch.Tensor):
                    return out["x"], i, "dict['x']"

                # 대안 키 탐색
                for key in ["encoder_out", "features", "last_hidden_state"]:
                    if key in out and isinstance(out[key], torch.Tensor):
                        return out[key], i, f"dict['{key}']"

                # 768-dim인 텐서 탐색
                for k, v in out.items():
                    if isinstance(v, torch.Tensor) and v.shape[-1] == EMBED_DIM:
                        return v, i, f"dict['{k}'] (dim=768 찾음)"

            elif isinstance(out, (tuple, list)):
                emb = next((r for r in out if isinstance(r, torch.Tensor)), None)
                if emb is not None:
                    return emb, i, "tuple[0]"

            elif isinstance(out, torch.Tensor):
                return out, i, "tensor"

        except TypeError as e:
            # features_only 인수를 지원하지 않는 경우
            if "features_only" in str(e) or "unexpected keyword" in str(e):
                continue
            raise
        except Exception as e:
            if i <= 3:
                print(f"    패턴 {i} 실패: {e}")
            continue

    return None, -1, "실패"


def embedding_stats(emb):
    """임베딩 통계 + 정상 판정."""
    arr = emb.detach().float().cpu().numpy()
    has_nan = bool(np.isnan(arr).any())
    has_inf = bool(np.isinf(arr).any())
    mean = float(np.nanmean(arr))
    std  = float(np.nanstd(arr))
    ok = not has_nan and not has_inf and 0.001 < std < 500
    return has_nan, has_inf, mean, std, ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("Pre-flight 2d: ECG-FM features_only + 0-fill forward 검증")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device} / GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"체크포인트: {args.checkpoint}")
    print()

    # ── 모델 로드 ────────────────────────────────────────────────────
    print("[1] 모델 로드")
    try:
        model = load_model(args.checkpoint, device)
        model.eval()
        total = sum(p.numel() for p in model.parameters())
        print(f"  {type(model).__name__}, {total/1e6:.1f}M params")
    except Exception as e:
        print(f"  [오류] 로드 실패: {e}")
        traceback.print_exc()
        return
    print()

    # ── forward 패턴 탐색 (12-lead 기준) ────────────────────────────
    print("[2] features_only forward 패턴 탐색 (12-lead 더미 신호)")
    signal = make_dummy_ecg()
    x12 = torch.tensor(signal).unsqueeze(0).to(device)  # (1, 12, 5000)
    emb12, pattern_idx, key_used = run_forward(model, x12)

    if emb12 is None:
        print("  [오류] 모든 패턴 실패. 에러 출력 전체를 확인하세요.")
        return

    has_nan, has_inf, mean, std, ok = embedding_stats(emb12)
    print(f"  성공 패턴: {pattern_idx}, 키: {key_used}")
    print(f"  shape: {tuple(emb12.shape)} — 예상: (T, 1, {EMBED_DIM})")
    print(f"  mean={mean:.4f}, std={std:.4f}, NaN={has_nan}, Inf={has_inf}")

    if emb12.shape[-1] != EMBED_DIM:
        print(f"  [주의] dim={emb12.shape[-1]} ≠ {EMBED_DIM}: 다른 출력 키를 얻은 것 같음.")
        print(f"         위 '출력 dict 키/shape' 목록을 확인하세요.")
    print()

    # ── 0-fill 다중 lead 테스트 ──────────────────────────────────────
    print("[3] 0-fill forward 테스트 (12→4→2→1→0 lead)")
    print("-" * 55)

    test_cases = [
        ("12-lead",                   list(range(12))),
        ("4-lead (I,II,V2,V5)+0fill", [0, 1, 7, 10]),
        ("2-lead (I,II)+0fill",       [0, 1]),
        ("1-lead (II)+0fill",         [1]),
        ("0-lead (극단)",              []),
    ]

    all_pass = True
    for name, available in test_cases:
        masked = apply_zero_mask(signal, available)
        x = torch.tensor(masked).unsqueeze(0).to(device)

        emb, _, _ = run_forward(model, x)
        if emb is None:
            print(f"  [오류] {name}: forward 실패")
            all_pass = False
            continue

        has_nan, has_inf, mean, std, ok = embedding_stats(emb)
        status = "" if ok else "[오류] "
        lead_str = f"[{','.join(LEAD_NAMES[i] for i in available)}]" if available else "[없음]"
        print(f"  {status} {name} {lead_str}")
        print(f"     shape={tuple(emb.shape)}, std={std:.4f}, NaN={has_nan}, Inf={has_inf}")
        if not ok:
            all_pass = False

    # ── 풀링 후 분류 헤드 진입 가능 여부 확인 ───────────────────────
    print()
    print("[4] 시간축 평균 풀링 → 분류 헤드 진입 형태 확인")
    if emb12 is not None and emb12.shape[-1] == EMBED_DIM:
        # (T, B, 768) → (B, 768)
        pooled = emb12.mean(dim=0)
        print(f"  emb12.mean(dim=0) → shape={tuple(pooled.shape)} (B=1, C=768)")
        print(f"  → 이 768-dim 벡터가 MLP 분류 헤드 입력이 됩니다.")
    else:
        print(f"  [주의] dim 불일치로 풀링 확인 스킵. 768-dim 출력 먼저 확보 필요.")

    # ── 최종 판정 ────────────────────────────────────────────────────
    print()
    print("=" * 65)
    if all_pass and emb12 is not None and emb12.shape[-1] == EMBED_DIM:
        print("Pre-flight 2 PASS")
        print("   - load_model_and_task(filename) 로드 성공")
        print(f"  - features_only 패턴 {pattern_idx} (키: {key_used}) 확정")
        print("   - 12→4→2→1→0 lead 0-fill 모두 정상 (NaN/Inf 없음)")
        print("   - 768-dim 풀링 → MLP 헤드 진입 형태 확인")
        print()
        print("   다음 단계:")
        print("   → decisions.md: Pre-flight 2 통과 기록")
        print("   → 단계 4 데이터 전처리 or 단계 5 베이스라인 학습 시작")
    else:
        print("[오류] 추가 조치 필요 — 아래 정보를 확인하세요:")
        if emb12 is not None and emb12.shape[-1] != EMBED_DIM:
            print(f"   - 출력 dim={emb12.shape[-1]} (768 아님): dict 키 목록 위에 출력됨")
        if not all_pass:
            print("   - 일부 lead 구성에서 NaN/Inf 발생")
    print("=" * 65)


if __name__ == "__main__":
    main()
