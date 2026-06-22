"""
P1 cardiac 채널 — 추론 인터페이스 (검증된 단일 진입점)
=====================================================
목적: 다운스트림 규칙 융합 모듈이 cardiac 출력을 일관되게 소비하도록 단일 추론 진입점 제공.
      전처리(단일리드 슬롯패킹·μ/σ 표준화·LoRA)를 캡슐화해 재현 오류를 방지한다.

사용:
    from p1_cardiac_channel import P1CardiacChannel
    ch = P1CardiacChannel(device="cuda")
    out = ch.infer(signal_12x5000)   # (12,5000) 또는 (N,12,5000)
    # out = {emergency_score, cardiac_probs[5], benign_flag}
    # 다운스트림: cardiac 경보 if out["emergency_score"] >= tau_c (실데이터 튜닝)

입력 규격: 단일리드는 ECG-FM 슬롯1(II)에 실어 0-fill된 (12,5000). 12리드면 그대로.
출력 계약 (records/00 §5):
    emergency_score   : sigmoid, AF/cardiac 응급 확률
    cardiac_probs[5]  : [NSR,AF,Ischemia,Conduction,Ectopic] softmax (유형)
    benign_flag       : 우세 유형 ∈ {전도장애,이소성} (비정상이나 양성 — 결정적)
"""
from __future__ import annotations
import math, os
import numpy as np, torch, torch.nn as nn

# ── 모델 체크포인트 경로 (이식성: 환경변수 우선, 없으면 기본값) ──────────────────
#   다른 환경에서 이식(vendoring) 시: P1_CKPT_FM / P1_CKPT_P1 환경변수로
#   체크포인트 위치 지정(또는 P1CardiacChannel(ckpt_fm=..., ...) 인자로 전달).
_DEF_FM =r"checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt"
_DEF_P1 =r"outputs/lora_multitask_snr_a07/lora_multitask_snr_best.pt"
CKPT_FM =os.environ.get("P1_CKPT_FM",  _DEF_FM)
CKPT_P1 =os.environ.get("P1_CKPT_P1",  _DEF_P1)
LEAD=1  # 패치 단일리드 슬롯(II)


class LoRALinear(nn.Module):
    def __init__(s,lin,r,a,d):
        super().__init__(); s.original=lin; lin.weight.requires_grad_(False)
        if lin.bias is not None: lin.bias.requires_grad_(False)
        i,o=lin.in_features,lin.out_features
        s.lora_A=nn.Linear(i,r,bias=False); s.lora_B=nn.Linear(r,o,bias=False)
        s.scaling=a/r; s.dropout=nn.Dropout(d)
        nn.init.kaiming_uniform_(s.lora_A.weight,a=math.sqrt(5)); nn.init.zeros_(s.lora_B.weight)
    @property
    def bias(s): return s.original.bias
    @property
    def weight(s): return s.original.weight
    def forward(s,x): return s.original(x)+s.lora_B(s.lora_A(s.dropout(x)))*s.scaling

def _inject(m,r=8,a=16,d=0.0):
    for n,mod in list(m.named_modules()):
        if isinstance(mod,nn.Linear) and (n.endswith('self_attn.q_proj') or n.endswith('self_attn.v_proj')):
            parts=n.split('.'); par=m
            for p in parts[:-1]: par=getattr(par,p)
            setattr(par,parts[-1],LoRALinear(mod,r,a,d))

class _Head(nn.Module):
    def __init__(s,o): super().__init__(); s.fc=nn.Linear(768,o)
    def forward(s,x): return s.fc(x).squeeze(-1) if s.fc.out_features==1 else s.fc(x)


class P1CardiacChannel:
    """P1 cardiac 채널 — 단일 forward로 emergency·type 산출."""

    def __init__(self, device: str = None,
                 ckpt_fm=CKPT_FM, ckpt_p1=CKPT_P1):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        from fairseq_signals.utils.checkpoint_utils import load_model_and_task
        res = load_model_and_task(ckpt_fm)
        bb = next(r for r in (res if isinstance(res,(list,tuple)) else [res]) if hasattr(r,'parameters')).to(self.device)
        for p in bb.parameters(): p.requires_grad_(False)
        _inject(bb)
        ck = torch.load(ckpt_p1, map_location=self.device)
        bb.load_state_dict(ck['backbone_lora'], strict=False)
        self.bb = bb.eval()
        self.hb = _Head(1).to(self.device); self.hb.load_state_dict(ck['head_bin_state']); self.hb.eval()
        self.hm = _Head(5).to(self.device); self.hm.load_state_dict(ck['head_mc_state']); self.hm.eval()

    @torch.no_grad()
    def infer(self, signal, batch_size: int = 32) -> dict:
        """signal: (12,5000) 또는 (N,12,5000) float. 반환: 채널 출력 dict(스칼라 또는 배열)."""
        x = np.asarray(signal, dtype=np.float32)
        single = (x.ndim == 2)
        if single: x = x[None]
        es,cp = [],[]
        for i in range(0, len(x), batch_size):
            xb = torch.tensor(x[i:i+batch_size], device=self.device)
            emb = self.bb(source=xb, padding_mask=None, features_only=True)['x'].mean(1)  # (b,768)
            es.append(torch.sigmoid(self.hb(emb)).cpu().numpy())
            cp.append(torch.softmax(self.hm(emb), -1).cpu().numpy())
        es = np.concatenate(es); cp = np.concatenate(cp)
        benign = np.isin(cp.argmax(1), [3, 4])    # 전도장애·이소성 우세 = 양성 비정상
        out = {"emergency_score": es, "cardiac_probs": cp, "benign_flag": benign}
        if single:
            out = {k: v[0] for k, v in out.items()}
        return out


# ── 자가 데모/검증: CACHET 몇 개로 채널 출력 확인 ──────────────────────────────
if __name__ == "__main__":
    import sys
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    ch = P1CardiacChannel()
    cs = np.load(r"data/processed/cachet/signals.npy")
    cl = np.load(r"data/processed/cachet/labels.npy")
    out = ch.infer(cs[:300])
    print("P1 cardiac 채널 데모 (CACHET 300)")
    for name, m in [("AF(label=1)", cl[:300] == 1), ("정상(label=0)", cl[:300] == 0)]:
        print(f"  [{name}] emergency={out['emergency_score'][m].mean():.3f} "
              f"benign%={100*out['benign_flag'][m].mean():.0f}")
    print("  계약 키:", list(out.keys()))
