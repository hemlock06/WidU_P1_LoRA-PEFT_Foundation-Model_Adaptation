"""
multi-SNR 모션 증강 모듈
=========================
목적:
  clean ECG에 NSTDB 모션 노이즈를 calibrated SNR로 주입해, ECG-FM이
  보지 못한 "모션 강건성"을 학습 단계에서 이식한다 (thesis 핵심 contribution).

설계 결정 (사용자 확정, decisions.md 2026-05-25):
  - SNR 분포 : 이산 집합 {24,18,12,6,0} dB, 균등 추출
  - clean 혼합: 샘플의 25%는 clean 유지 (p_noise=0.75)
  - lead별 SNR: 각 lead 독립 샘플링 (wearable lead별 접촉 품질 차이 모사)

핵심 수식 (4페이지_리빌딩_최종정리.md 섹션 6):
  SNR_dB = 10·log10(P_signal / P_noise)
  목표 SNR을 맞추는 노이즈 스케일 계수:
    alpha = sqrt( P_signal / (P_noise_raw · 10^(SNR/10)) )
    x_noisy = x + alpha · n
  → alpha는 신호 자기 파워로 보정되므로 절대 단위에 무관 (unit-invariant).

파이프라인 위치 (스펙 8-5):
  clean → [이 모듈: 노이즈 주입] → RLM 마스킹 → ECG-FM+LoRA
  노이즈를 먼저, 마스킹을 나중에 → 각 lead가 {노이즈 신호} 또는 {0=부재}로
  깔끔히 분리되어 multi-SNR(present-but-noisy)과 RLM(absent) 역할 경계 유지.

노이즈 종류 (NSTDB):
  bw = baseline wander, ma = muscle artifact,
  em = electrode motion (가장 까다로움 → 기본 가중치 강조)

노이즈 합성 모드 (noise_mode):
  - 'single' (기본): 리드마다 bw/em/ma 중 1종만 선택해 주입 — 리드당 1종 단순화.
  - 'mixed'        : 리드마다 bw·em·ma '각각 독립 구간'을 std 정규화 후 noise_weights로
                     가중합해 '동시 중첩' 노이즈를 만들어 목표 SNR로 1회 주입.
                     실 ECG 노이즈는 세 성분이 한 시점에 중첩되므로 더 사실적.
  두 모드는 동일 _add_at_snr(합성 노이즈 전체 파워 기준 SNR 보정)로 주입되어
  주입 후 실측 SNR이 목표와 정확히 일치한다(unit-invariant).
"""

import os

import numpy as np
import torch
from scipy.signal import resample_poly

NSTDB_DIR_DEFAULT = "data/raw/nstdb"
NOISE_TYPES = ("bw", "em", "ma")
NSTDB_FS = 360            # NSTDB 원본 샘플링레이트
TARGET_FS = 500           # ECG-FM 입력 샘플링레이트
# 500/360 = 25/18 (gcd=20) → resample_poly(up=25, down=18)
_UP, _DOWN = 25, 18
_EPS = 1e-8


class MultiSNRNoise:
    """
    NSTDB 노이즈를 미리 500Hz로 리샘플해 메모리에 적재하고,
    배치 텐서에 per-sample·per-lead 노이즈를 주입한다.
    """

    def __init__(
        self,
        nstdb_dir: str = NSTDB_DIR_DEFAULT,
        snr_set=(24, 18, 12, 6, 0),
        noise_weights=(0.25, 0.50, 0.25),  # (bw, em, ma) — em 강조
        device: torch.device = torch.device("cpu"),
        seed: int = 42,
        noise_mode: str = "single",   # 'single' | 'mixed' | 'mixed_temporal'
    ):
        assert len(noise_weights) == len(NOISE_TYPES)
        assert noise_mode in ("single", "mixed", "mixed_temporal"), f"noise_mode={noise_mode}"
        self.snr_set = np.asarray(snr_set, dtype=np.float64)
        self.noise_weights = np.asarray(noise_weights, dtype=np.float64)
        self.noise_weights /= self.noise_weights.sum()
        self.device = device
        self.noise_mode = noise_mode
        self.rng = np.random.default_rng(seed)

        # 노이즈 레코드를 500Hz로 한 번만 리샘플해 1D pool로 적재
        self.noise_pool = {}     # type -> torch.Tensor (1D, device)
        for t in NOISE_TYPES:
            self.noise_pool[t] = self._load_noise(nstdb_dir, t)

    def _load_noise(self, nstdb_dir: str, noise_type: str) -> torch.Tensor:
        import wfdb
        path = os.path.join(nstdb_dir, noise_type)
        sig, _ = wfdb.rdsamp(path)              # (650000, 2) @360Hz
        res = resample_poly(sig, _UP, _DOWN, axis=0)   # (~902777, 2) @500Hz
        # 2채널을 이어붙여 하나의 긴 1D pool로 → 랜덤 슬라이스 다양성 확보
        flat = res.T.reshape(-1).astype(np.float32)
        return torch.from_numpy(flat).to(self.device)

    def to(self, device: torch.device):
        self.device = device
        for t in self.noise_pool:
            self.noise_pool[t] = self.noise_pool[t].to(device)
        return self

    # ── 학습용: per-sample 게이트 + per-lead 독립 SNR ──────────────────
    def inject(self, x: torch.Tensor, p_noise: float = 0.75,
               noise_mode: str = None) -> torch.Tensor:
        """
        x: (B, C, T) clean 배치 (C=12, T=5000)
        반환: 같은 shape의 노이즈 주입 배치.
        - 각 샘플은 p_noise 확률로만 노이즈 적용 (나머지는 clean 유지)
        - 노이즈 적용 샘플의 각 lead는 독립적으로 SNR·구간 선택
        - noise_mode='single': lead마다 노이즈 1종 선택
          noise_mode='mixed' : lead마다 bw·em·ma 가중합성(동시 중첩) 1회 주입
          (None이면 인스턴스 기본값 self.noise_mode 사용)
        """
        mode = noise_mode or self.noise_mode
        B, C, T = x.shape
        out = x.clone()
        for b in range(B):
            if self.rng.random() >= p_noise:
                continue  # clean 유지
            for c in range(C):
                snr = float(self.rng.choice(self.snr_set))
                if mode == "mixed":
                    n = self._mixed_segment(T)
                elif mode == "mixed_temporal":
                    n = self._mixed_segment(T, temporal=True)
                else:
                    ntype = NOISE_TYPES[self.rng.choice(len(NOISE_TYPES), p=self.noise_weights)]
                    n = self._random_segment(ntype, T)
                out[b, c] = self._add_at_snr(out[b, c], n, snr)
        return out

    # ── 평가용: 전 lead에 고정 SNR 주입 (단계 8 SNR 저하 곡선) ──────────
    def inject_fixed(self, x: torch.Tensor, snr_db: float,
                     noise_mode: str = None) -> torch.Tensor:
        """
        모든 샘플의 모든 lead에 동일 SNR을 주입 (노이즈 종류·구간은 랜덤).
        SNR 저하 곡선 평가 전용 — 학습에는 사용하지 않음.
        noise_mode 의미는 inject()와 동일 (None이면 self.noise_mode).
        """
        mode = noise_mode or self.noise_mode
        B, C, T = x.shape
        out = x.clone()
        for b in range(B):
            for c in range(C):
                if mode == "mixed":
                    n = self._mixed_segment(T)
                elif mode == "mixed_temporal":
                    n = self._mixed_segment(T, temporal=True)
                else:
                    ntype = NOISE_TYPES[self.rng.choice(len(NOISE_TYPES), p=self.noise_weights)]
                    n = self._random_segment(ntype, T)
                out[b, c] = self._add_at_snr(out[b, c], n, float(snr_db))
        return out

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────
    def _random_segment(self, noise_type: str, length: int) -> torch.Tensor:
        pool = self.noise_pool[noise_type]
        start = int(self.rng.integers(0, pool.shape[0] - length))
        return pool[start:start + length]

    def _mixed_segment(self, length: int, temporal: bool = False) -> torch.Tensor:
        """
        bw·em·ma 각각 '독립 구간'을 뽑아 std로 정규화한 뒤 noise_weights로 가중합한
        합성 노이즈(동시 중첩)를 반환. 정규화 없이 합치면 파워 큰 종류가 지배하므로
        종류별 단위분산 정렬 후 가중합 → 가중치가 실제 기여 비율을 결정한다.
        최종 절대 스케일은 _add_at_snr가 목표 SNR로 재보정하므로 무관(unit-invariant).
        temporal=True(mixed_temporal): std 정규화 '후' 종류별 시간 엔벨로프를 곱해 노이즈
        에너지의 시간 분포를 현실화(bw 지속·ma 빈번·em burst). _add_at_snr가 전체 파워로
        재보정하므로 '평균 SNR'은 mixed와 동일, 차이는 에너지의 시간 분포뿐(공정 비교의 핵심).
        """
        mix = None
        for i, t in enumerate(NOISE_TYPES):           # (bw, em, ma) 순서 = noise_weights 순서
            seg = self._random_segment(t, length)
            seg = seg / (seg.std() + _EPS)            # std 정규화 (단위분산)
            if temporal:
                seg = seg * self._temporal_envelope(t, length, self.rng)  # 시간 게이팅(정규화 후)
            w = float(self.noise_weights[i])
            mix = w * seg if mix is None else mix + w * seg
        return mix

    def _temporal_envelope(self, noise_type: str, length: int, rng) -> torch.Tensor:
        """
        종류별 시간 게이트 e(t)∈[0,1] (리드별 독립, 물리 직관 고정값 — 성능 보고 튜닝 금지):
          bw : 전 구간 1.0          (호흡성 baseline wander = 지속적)
          ma : 듀티 ~50%, on/off 블록 0.3~0.7s 교대 (근전도 = 빈번·짧게)
          em : 10초에 1~2회, 각 0.5~1.5s burst만 1 (전극 움직임 = 드문 큰 burst)
        on 블록 경계에 25샘플(50ms) 코사인 ramp로 클릭 방지.
        """
        fs = TARGET_FS
        env = np.zeros(length, dtype=np.float32)
        ramp = 25

        def block(s, e):
            s = max(0, int(s)); e = min(length, int(e))
            if e <= s:
                return
            env[s:e] = 1.0
            r = min(ramp, (e - s) // 2)
            if r > 0:
                up = (0.5 * (1 - np.cos(np.linspace(0, np.pi, r)))).astype(np.float32)
                env[s:s + r] *= up
                env[e - r:e] *= up[::-1]

        if noise_type == "bw":
            env[:] = 1.0
        elif noise_type == "ma":
            t, on = 0, bool(rng.integers(0, 2))       # 시작 상태 랜덤
            while t < length:
                blk = int(rng.uniform(0.3, 0.7) * fs)
                if on:
                    block(t, t + blk)
                t += blk
                on = not on
        elif noise_type == "em":
            for _ in range(int(rng.integers(1, 3))):  # 1~2회 burst
                blk = int(rng.uniform(0.5, 1.5) * fs)
                s = int(rng.integers(0, max(1, length - blk)))
                block(s, s + blk)
        else:
            env[:] = 1.0
        return torch.from_numpy(env).to(self.device)

    @staticmethod
    def _add_at_snr(sig: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
        p_sig = torch.mean(sig * sig)
        p_noise = torch.mean(noise * noise)
        if p_sig < _EPS or p_noise < _EPS:
            return sig  # flat lead → alpha 폭발 방지, 주입 생략
        alpha = torch.sqrt(p_sig / (p_noise * (10.0 ** (snr_db / 10.0))))
        return sig + alpha * noise


# ── 단독 실행: 모듈 자체 검증 (smoke test) ────────────────────────────
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={dev}")

    # 가짜 clean 배치 (모드 간 동일) — std≈13 (CPSC 유사 스케일)
    torch.manual_seed(0)
    x = torch.randn(4, 12, 5000, device=dev) * 13.0
    targets = [24, 18, 12, 6, 0, -6]
    TOL = 0.05          # 실측-목표 허용 오차 (dB); _add_at_snr는 해석적으로 정확
    ok_all = True

    for mode in ("single", "mixed", "mixed_temporal"):
        aug = MultiSNRNoise(device=dev, noise_mode=mode, seed=42)
        if mode == "single":
            print("[smoke] 노이즈 pool 길이: "
                  + ", ".join(f"{t}={aug.noise_pool[t].shape[0]:,}" for t in NOISE_TYPES))
        print(f"\n=== [{mode}] 고정 SNR 주입 후 실측 SNR ===")
        print(f"{'목표SNR(dB)':>10} {'실측SNR(dB)':>10} {'오차':>8} {'판정':>5}")
        for target in targets:
            noisy = aug.inject_fixed(x, target)
            diff = noisy - x
            meas = 10 * torch.log10((x ** 2).mean() / (diff ** 2).mean()).item()
            err = abs(meas - target)
            ok = err < TOL and not torch.isnan(noisy).any() and not torch.isinf(noisy).any()
            ok_all &= ok
            print(f"{target:>10} {meas:>10.3f} {err:>8.4f} {'OK' if ok else 'FAIL':>5}")

        # 학습용 inject: clean 유지 비율 + NaN/Inf
        noisy = aug.inject(x, p_noise=0.75)
        n_clean = sum(torch.allclose(noisy[b], x[b]) for b in range(x.shape[0]))
        nan_inf = bool(torch.isnan(noisy).any() or torch.isinf(noisy).any())
        ok_all &= (not nan_inf)
        print(f"[{mode}] inject(p_noise=0.75): 배치 4개 중 clean 유지 {n_clean}개 | "
              f"NaN/Inf={nan_inf}")

    # mixed 전용: 합성 노이즈가 3종을 실제로 섞는지(단일종과 구별) 확인
    augm = MultiSNRNoise(device=dev, noise_mode="mixed", seed=7)
    nmix = augm._mixed_segment(5000)
    print(f"\n[mixed 합성 점검] n_mix std={nmix.std().item():.4f} "
          f"(가중 std 합 이론치≈{float((augm.noise_weights**2).sum())**0.5:.4f}), "
          f"len={nmix.shape[0]}")

    # mixed_temporal 엔벨로프가 종류별로 실제 시간 뭉침을 만드는지 (bw 지속 vs em burst)
    augt = MultiSNRNoise(device=dev, noise_mode="mixed_temporal", seed=7)
    print("\n[mixed_temporal 엔벨로프] on-비율(e>0.5) — bw≈1.0 지속 / ma≈0.5 빈번 / em 낮음(드문 burst)")
    on_fracs = {}
    for t in ("bw", "ma", "em"):
        es = [augt._temporal_envelope(t, 5000, augt.rng).cpu().numpy() for _ in range(30)]
        on_fracs[t] = float(np.mean([(e > 0.5).mean() for e in es]))
        print(f"   {t}: on-비율={on_fracs[t]:.2f}")
    # 정성 검증: bw 지속(>0.95) > ma 중간 > em 집중(<0.30)  → 시간 구조 차등 확인
    env_ok = on_fracs["bw"] > 0.95 and on_fracs["em"] < 0.30 and on_fracs["ma"] > on_fracs["em"]
    ok_all &= env_ok
    print(f"   엔벨로프 시간뭉침 차등: {'OK' if env_ok else 'FAIL'}")

    print(f"\n[smoke] {'전체 통과 ✓' if ok_all else '실패 ✗ — 확인 필요'}")
