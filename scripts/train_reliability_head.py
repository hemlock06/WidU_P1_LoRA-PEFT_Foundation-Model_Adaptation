"""
전용 단일리드 reliability 헤드 재학습 (NSTDB 자가라벨)
=======================================================
동기: 기존 게이트(PhysioNet2011·12리드 학습)는 단일리드 깨끗-vs-모션 분리가 약함(AUROC 0.68)
      → Platt 보정도 캡. 전용 헤드를 올바른 라벨로 학습해 천장 돌파 시도.
설계: ECG-FM(frozen)+P1 LoRA mean-pool(768) → 작은 MLP(768→64→1). 백본 동결, 헤드만 학습.
자가라벨: 깨끗 단일리드=0(신뢰), NSTDB 모션주입 단일리드=1(불신뢰). SNR {12,6,0,-6} 범위.
정직성 검증: NSTDB로 학습하되 실 모션(CACHET·PTT 운동)에 일반화되는지 — NSTDB 과적합 여부.
"""
from __future__ import annotations
import math, os, sys, random, json
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from multisnr import MultiSNRNoise
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from sklearn.metrics import roc_auc_score

CKPT_FM=r"D:\WidU_ecg-fm_emergency-detection\checkpoints\ecg-fm\mimic_iv_ecg_physionet_pretrained.pt"
CKPT_P1=r"D:\WidU_ecg-fm_emergency-detection\outputs\lora_multitask_snr_a07\lora_multitask_snr_best.pt"
CPSC_TR=r"D:\WidU_ecg-fm_emergency-detection\data\processed\cpsc2018_mc\train"
CPSC_TE=r"D:\WidU_ecg-fm_emergency-detection\data\processed\cpsc2018_mc\test"
CACHET=r"D:\WidU_ecg-fm_emergency-detection\data\processed\cachet"
NSTDB=r"D:\WidU_ecg-fm_emergency-detection\data\raw\nstdb"
PTT=r"D:\WidU_multimodal_fusion\raw\ptt_ppg"
OUT=r"D:\WidU_ecg-fm_emergency-detection\outputs\gate\reliability_head.pt"
LEAD=1; LAM=0.6; SNRS=[12,6,0,-6]

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
class RelHead(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(768,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(0.3),nn.Linear(64,1))
    def forward(s,x): return s.net(x).squeeze(-1)

def load(dev):
    from fairseq_signals.utils.checkpoint_utils import load_model_and_task
    res=load_model_and_task(CKPT_FM)
    bb=next(r for r in (res if isinstance(res,(list,tuple)) else [res]) if hasattr(r,'parameters')).to(dev)
    for p in bb.parameters(): p.requires_grad_(False)
    inject(bb); ck=torch.load(CKPT_P1,map_location=dev); bb.load_state_dict(ck['backbone_lora'],strict=False)
    hb=Head(1).to(dev); hb.load_state_dict(ck['head_bin_state']); bb.eval(); hb.eval()
    return bb,hb

def single(s12):
    o=np.zeros_like(s12); o[LEAD,:]=s12[LEAD,:]; return o

@torch.no_grad()
def feats(bb,x,dev,bs=32):
    out=[]
    for i in range(0,len(x),bs):
        xb=torch.tensor(x[i:i+bs],dtype=torch.float32,device=dev)
        out.append(bb(source=xb,padding_mask=None,features_only=True)['x'].mean(1).cpu().numpy())
    return np.concatenate(out)
@torch.no_grad()
def emerg(bb,hb,x,dev,bs=32):
    out=[]
    for i in range(0,len(x),bs):
        xb=torch.tensor(x[i:i+bs],dtype=torch.float32,device=dev)
        out.append(torch.sigmoid(hb(bb(source=xb,padding_mask=None,features_only=True)['x'].mean(1))).cpu().numpy())
    return np.concatenate(out)

def build(bb,sigs,aug,dev,rng):
    """깨끗(0) + 모션(1) 단일리드 특징 생성."""
    clean=np.stack([single(s) for s in sigs])
    snr=rng.choice(SNRS,size=len(clean))
    motion=np.stack([aug.inject_fixed(torch.tensor(clean[i:i+1],dtype=torch.float32,device=dev),float(snr[i])).cpu().numpy()[0] for i in range(len(clean))])
    Fc=feats(bb,clean,dev); Fm=feats(bb,motion,dev)
    X=np.r_[Fc,Fm]; y=np.r_[np.zeros(len(Fc)),np.ones(len(Fm))]
    return X.astype(np.float32),y.astype(np.float32)

def main():
    set_det(42); dev='cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev} | 전용 reliability 헤드 학습 | λ={LAM}")
    bb,hb=load(dev); aug=MultiSNRNoise(nstdb_dir=NSTDB,device=dev,seed=42)

    tr=np.load(os.path.join(CPSC_TR,'signals.npy'))
    te=np.load(os.path.join(CPSC_TE,'signals.npy')); telab=np.load(os.path.join(CPSC_TE,'labels.npy'))
    itr=np.random.RandomState(0).permutation(len(tr))[:700]
    print("특징 추출(train)..."); Xtr,ytr=build(bb,tr[itr],aug,dev,np.random.RandomState(1))
    print("특징 추출(test held-out)...")
    ite=np.random.RandomState(0).permutation(len(te))[:200]
    Xte,yte=build(bb,te[ite],aug,dev,np.random.RandomState(3))

    # 표준화
    mu,sd=Xtr.mean(0),Xtr.std(0)+1e-6
    Xtr_n=(Xtr-mu)/sd; Xte_n=(Xte-mu)/sd
    head=RelHead().to(dev)
    opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-3)
    bce=nn.BCEWithLogitsLoss()
    Xt=torch.tensor(Xtr_n,device=dev); Yt=torch.tensor(ytr,device=dev)
    print("\n학습:")
    for ep in range(1,41):
        head.train(); perm=torch.randperm(len(Xt),device=dev)
        for i in range(0,len(Xt),128):
            b=perm[i:i+128]; opt.zero_grad()
            loss=bce(head(Xt[b]),Yt[b]); loss.backward(); opt.step()
        if ep%10==0:
            head.eval()
            with torch.no_grad():
                ptr=torch.sigmoid(head(Xt)).cpu().numpy()
                pte=torch.sigmoid(head(torch.tensor(Xte_n,device=dev))).cpu().numpy()
            print(f"  ep{ep:2d} loss={loss.item():.3f} | train AUROC={roc_auc_score(ytr,ptr):.4f} | held-out AUROC={roc_auc_score(yte,pte):.4f}")

    head.eval()
    def rel(sig12_batch):  # (N,12,5000) → reliability
        F=feats(bb,sig12_batch,dev); Fn=(F-mu)/sd
        with torch.no_grad(): return torch.sigmoid(head(torch.tensor(Fn,device=dev))).cpu().numpy()

    # ── 검증1: held-out CPSC NSR 단일리드 per-SNR ──
    print("\n"+"="*60+"\n[검증1] held-out CPSC NSR — 전용헤드 reliability per-SNR\n"+"="*60)
    nsr=np.stack([single(s) for s in te[telab==0][:150]])
    print(f"  {'조건':>8} {'rel평균':>8} {'<0.3비율':>9} {'>0.7비율':>9}")
    for cond in ['clean',12,6,0,-6]:
        x=nsr if cond=='clean' else np.stack([aug.inject_fixed(torch.tensor(nsr[i:i+1],dtype=torch.float32,device=dev),float(cond)).cpu().numpy()[0] for i in range(len(nsr))])
        r=rel(x)
        print(f"  {str(cond):>8} {r.mean():>8.3f} {100*(r<0.3).mean():>8.1f}% {100*(r>0.7).mean():>8.1f}%")

    # ── 검증2: CACHET 실 웨어러블 — NSTDB 과적합 여부 + AF 보존 ──
    print("\n"+"="*60+"\n[검증2] CACHET 실 웨어러블 (NSTDB 비포함 실모션) — 일반화\n"+"="*60)
    cs=np.load(os.path.join(CACHET,'signals.npy')); cl=np.load(os.path.join(CACHET,'labels.npy'))
    es=emerg(bb,hb,cs,dev); r=rel(cs); eff=es*(1-LAM*r)
    print(f"  reliability 평균={r.mean():.3f} (<0.3:{100*(r<0.3).mean():.0f}% >0.7:{100*(r>0.7).mean():.0f}%)")
    print(f"  AF effective_cardiac AUROC={roc_auc_score(cl,eff):.4f} | AF eff평균={eff[cl==1].mean():.3f}")

    # ── 검증3: PTT-PPG 운동 (실 운동모션, NSTDB 아님) ──
    print("\n"+"="*60+"\n[검증3] PTT-PPG 운동 (실 운동모션) — 일반화\n"+"="*60)
    import wfdb
    subs=sorted(set(f.split('_')[0] for f in os.listdir(PTT) if f.endswith('.hea')))[:12]
    for act in ['sit','run']:
        E,R=[],[]
        for sub in subs:
            rp=os.path.join(PTT,f"{sub}_{act}")
            if not os.path.exists(rp+'.hea'): continue
            ecg=wfdb.rdrecord(rp,channels=[0]).p_signal[:,0].astype(np.float32)
            ws=np.stack([np.r_[np.zeros((LEAD,5000)),ecg[st:st+5000][None],np.zeros((11-LEAD,5000))] for st in np.linspace(15000,len(ecg)-5000,8).astype(int)])
            e=emerg(bb,hb,ws,dev); r=rel(ws)
            E.append(e.mean()); R.append(r.mean())
        E=np.array(E); R=np.array(R); eff=E*(1-LAM*R)
        print(f"  {act:>5}: emergency={E.mean():.3f} 전용rel={R.mean():.3f} → effective={eff.mean():.3f}")

    torch.save({"head_state":head.state_dict(),"mu":mu,"sd":sd,"lambda":LAM,"lead":LEAD},OUT)
    print(f"\n[저장] 전용 reliability 헤드 → {OUT}")

if __name__=="__main__": main()
