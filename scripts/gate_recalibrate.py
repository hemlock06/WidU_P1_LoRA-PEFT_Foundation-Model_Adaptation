"""
게이트 단일리드 웨어러블 재보정 (#1 블로커 해소)
==================================================
문제(척추 실측): PhysioNet2011·12리드로 학습된 reliability 게이트가 단일리드 웨어러블에서
  깨끗한 신호도 60~66% alert(rel~0.6)로 과비관 → 진짜 AF도 억제 위험.
목표: reliability를 단조 연속 보정 — 깨끗 단일리드≈낮게, 모션≈높게.
방법: 지도 Platt(affine) 재보정. 앵커 = 깨끗 단일리드 CPSC(신뢰=0) vs NSTDB 모션주입(불신뢰=1).
      게이트 raw logit은 동결, 그 위 보정맵만 학습(순위 보존, baseline 시프트 교정).
결합기(A 연속 소프트): effective_cardiac = emergency_score × (1 − λ·rel_calibrated), λ=0.6.
"""
from __future__ import annotations
import math, os, sys, random, json
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from multisnr import MultiSNRNoise
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

CKPT_FM=r"D:\WidU_ecg-fm_emergency-detection\checkpoints\ecg-fm\mimic_iv_ecg_physionet_pretrained.pt"
CKPT_P1=r"D:\WidU_ecg-fm_emergency-detection\outputs\lora_multitask_snr_a07\lora_multitask_snr_best.pt"
CKPT_GATE=r"D:\WidU_ecg-fm_emergency-detection\outputs\gate\gate_best.pt"
CPSC_TR=r"D:\WidU_ecg-fm_emergency-detection\data\processed\cpsc2018_mc\train"
CPSC_TE=r"D:\WidU_ecg-fm_emergency-detection\data\processed\cpsc2018_mc\test"
CACHET=r"D:\WidU_ecg-fm_emergency-detection\data\processed\cachet"
NSTDB=r"D:\WidU_ecg-fm_emergency-detection\data\raw\nstdb"
PTT=r"D:\WidU_multimodal_fusion\raw\ptt_ppg"
OUT=r"D:\WidU_ecg-fm_emergency-detection\outputs\gate\recalib.json"
LEAD=1; T_ALERT=0.4753; LAM=0.6

def set_det(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

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
def inject(m,r=8,a=16,d=0.0):
    for n,mod in list(m.named_modules()):
        if isinstance(mod,nn.Linear) and (n.endswith('self_attn.q_proj') or n.endswith('self_attn.v_proj')):
            parts=n.split('.'); par=m
            for p in parts[:-1]: par=getattr(par,p)
            setattr(par,parts[-1],LoRALinear(mod,r,a,d))
class Head(nn.Module):
    def __init__(s,o): super().__init__(); s.fc=nn.Linear(768,o)
    def forward(s,x): return s.fc(x).squeeze(-1) if s.fc.out_features==1 else s.fc(x)

def load(dev):
    from fairseq_signals.utils.checkpoint_utils import load_model_and_task
    res=load_model_and_task(CKPT_FM)
    bb=next(r for r in (res if isinstance(res,(list,tuple)) else [res]) if hasattr(r,'parameters')).to(dev)
    for p in bb.parameters(): p.requires_grad_(False)
    inject(bb); ck=torch.load(CKPT_P1,map_location=dev); bb.load_state_dict(ck['backbone_lora'],strict=False)
    hb=Head(1).to(dev); hb.load_state_dict(ck['head_bin_state'])
    gk=torch.load(CKPT_GATE,map_location=dev); hg=Head(1).to(dev); hg.load_state_dict(gk['head_state'])
    bb.eval(); hb.eval(); hg.eval(); return bb,hb,hg

@torch.no_grad()
def gate_logits(bb,hg,x,dev,bs=32):
    out=[]
    for i in range(0,len(x),bs):
        xb=torch.tensor(x[i:i+bs],dtype=torch.float32,device=dev)
        emb=bb(source=xb,padding_mask=None,features_only=True)['x'].mean(1)
        out.append(hg(emb).cpu().numpy())
    return np.concatenate(out)
@torch.no_grad()
def emerg(bb,hb,x,dev,bs=32):
    out=[]
    for i in range(0,len(x),bs):
        xb=torch.tensor(x[i:i+bs],dtype=torch.float32,device=dev)
        emb=bb(source=xb,padding_mask=None,features_only=True)['x'].mean(1)
        out.append(torch.sigmoid(hb(emb)).cpu().numpy())
    return np.concatenate(out)

def single(sig12):
    o=np.zeros_like(sig12); o[LEAD,:]=sig12[LEAD,:]; return o

def main():
    set_det(42); dev='cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev} | 게이트 단일리드 재보정 (Platt) | λ={LAM}")
    bb,hb,hg=load(dev)
    aug=MultiSNRNoise(nstdb_dir=NSTDB,device=dev,seed=42)

    # ── 보정셋: CPSC train 단일리드 깨끗(0) vs 모션(1) ──
    tr=np.load(os.path.join(CPSC_TR,'signals.npy'))
    idx=np.random.RandomState(0).permutation(len(tr))[:300]
    clean=np.stack([single(tr[i]) for i in idx])
    # 모션: SNR을 {6,0,-6} 균등 주입
    snrs=np.random.RandomState(1).choice([6,0,-6],size=len(clean))
    motion=np.stack([aug.inject_fixed(torch.tensor(clean[i:i+1],dtype=torch.float32,device=dev),float(snrs[i])).cpu().numpy()[0] for i in range(len(clean))])
    Xc=gate_logits(bb,hg,clean,dev); Xm=gate_logits(bb,hg,motion,dev)
    print(f"\n[raw 게이트 logit 분리도] 깨끗 mean={Xc.mean():.2f} | 모션 mean={Xm.mean():.2f} | AUROC={roc_auc_score([0]*len(Xc)+[1]*len(Xm),np.r_[Xc,Xm]):.4f}")
    print(f"  (현 sigmoid) 깨끗 rel={1/(1+np.exp(-Xc)).mean():.3f} | 모션 rel={1/(1+np.exp(-Xm)).mean():.3f}")

    # ── Platt 적합 (train/test 분리) ──
    L=np.r_[Xc,Xm].reshape(-1,1); y=np.r_[np.zeros(len(Xc)),np.ones(len(Xm))]
    sp=np.random.RandomState(2).permutation(len(y)); cut=int(0.7*len(y))
    tri,tei=sp[:cut],sp[cut:]
    platt=LogisticRegression(C=1.0).fit(L[tri],y[tri])
    a,b=float(platt.coef_[0,0]),float(platt.intercept_[0])
    def recal(logit): return 1/(1+np.exp(-(a*logit+b)))
    # 검증(held-out)
    rc_clean=recal(Xc[tei[y[tei]==0] - (0 if True else 0)]) if False else None
    # 단순 평가: held-out에서 깨끗/모션 보정 rel
    te_clean=recal(np.r_[Xc,Xm][tei][y[tei]==0]); te_mot=recal(np.r_[Xc,Xm][tei][y[tei]==1])
    print(f"\n[Platt 보정맵] rel_cal = sigmoid({a:.3f}·logit + {b:.3f})")
    print(f"  held-out 깨끗 rel_cal={te_clean.mean():.3f} (목표 낮음) | 모션 rel_cal={te_mot.mean():.3f} (목표 높음)")

    # ── 독립 검증 1: CPSC test 단일리드 깨끗 vs 모션 ──
    te=np.load(os.path.join(CPSC_TE,'signals.npy')); lab=np.load(os.path.join(CPSC_TE,'labels.npy'))
    nsr=np.stack([single(s) for s in te[lab==0][:150]])
    print("\n"+"="*60+"\n[검증1] CPSC test NSR 단일리드 — 보정 전/후\n"+"="*60)
    print(f"  {'조건':>8} {'기존rel':>8} {'보정rel':>8} {'기존alert%':>10} {'보정<0.3%':>10}")
    for cond in ['clean',6,0,-6]:
        x=nsr if cond=='clean' else np.stack([aug.inject_fixed(torch.tensor(nsr[i:i+1],dtype=torch.float32,device=dev),float(cond)).cpu().numpy()[0] for i in range(len(nsr))])
        lg=gate_logits(bb,hg,x,dev); old=1/(1+np.exp(-lg)); new=recal(lg)
        print(f"  {str(cond):>8} {old.mean():>8.3f} {new.mean():>8.3f} {100*(old>=T_ALERT).mean():>9.1f}% {100*(new<0.3).mean():>9.1f}%")

    # ── 독립 검증 2: CACHET (실 웨어러블 AF) — 결합기 효과 ──
    print("\n"+"="*60+"\n[검증2] CACHET 실 웨어러블 — AF 보존 (결합기 A)\n"+"="*60)
    cs=np.load(os.path.join(CACHET,'signals.npy')); cl=np.load(os.path.join(CACHET,'labels.npy'))
    es=emerg(bb,hb,cs,dev); lg=gate_logits(bb,hg,cs,dev); old=1/(1+np.exp(-lg)); new=recal(lg)
    eff_old=es*(1-LAM*old); eff_new=es*(1-LAM*new)
    print(f"  기존 rel={old.mean():.3f}(alert {100*(old>=T_ALERT).mean():.0f}%) → 보정 rel={new.mean():.3f}")
    print(f"  AF 윈도우 effective_cardiac: 기존={eff_old[cl==1].mean():.3f} → 보정={eff_new[cl==1].mean():.3f} (↑=AF 더 보존)")
    print(f"  AF AUROC(effective): 기존={roc_auc_score(cl,eff_old):.4f} → 보정={roc_auc_score(cl,eff_new):.4f}")

    # ── 독립 검증 3: PTT-PPG 운동 — 오경보 억제 유지 ──
    print("\n"+"="*60+"\n[검증3] PTT-PPG 운동 — 모션 억제 유지\n"+"="*60)
    import wfdb
    subs=sorted(set(f.split('_')[0] for f in os.listdir(PTT) if f.endswith('.hea')))[:12]
    for act in ['sit','run']:
        E,R=[],[]
        for sub in subs:
            rp=os.path.join(PTT,f"{sub}_{act}")
            if not os.path.exists(rp+'.hea'): continue
            ecg=wfdb.rdrecord(rp,channels=[0]).p_signal[:,0].astype(np.float32)
            ws=[]
            for st in np.linspace(15000,len(ecg)-5000,8).astype(int):
                seg=np.zeros((12,5000),dtype=np.float32); seg[LEAD]=ecg[st:st+5000]; ws.append(seg)
            ws=np.stack(ws); e=emerg(bb,hb,ws,dev); lg=gate_logits(bb,hg,ws,dev); new=recal(lg)
            E.append(e.mean()); R.append(new.mean())
        E=np.array(E); R=np.array(R); eff=E*(1-LAM*R)
        print(f"  {act:>5}: emergency={E.mean():.3f} 보정rel={R.mean():.3f} → effective={eff.mean():.3f} (운동은 낮아야 억제)")

    json.dump({"platt_a":a,"platt_b":b,"lambda":LAM,"lead":LEAD,
               "note":"rel_cal=sigmoid(a*gate_logit+b); effective=emergency*(1-lambda*rel_cal)"},
              open(OUT,'w'),ensure_ascii=False,indent=2)
    print(f"\n[저장] 보정 파라미터 → {OUT}")

if __name__=="__main__": main()
