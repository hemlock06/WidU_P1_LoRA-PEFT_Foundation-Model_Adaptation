"""
Stage 0 척추 검증 — 공개데이터로 웨어러블 단일리드 적용 타당성 분석
====================================================================
목적: 실세계 적용 전, 공개데이터로 단일리드 적용 3축을 실측한다.
  척추1 CACHET   : 패치형 단일리드에서 AF 전이되는가 (운영점 포함)
  척추2 NSTDB    : 모션 노이즈를 ECG 비정상으로 오인 안 하는가
  척추3 PTT-PPG  : 운동(동성빈맥)에 cardiac 오발화 안 하는가 (실 ECG+IMU 동시수집)

모두 단일리드(slot1=II) 패치 시나리오로 통일. 결정성 고정(seed=42).
"""
from __future__ import annotations
import math, os, sys, random
import numpy as np, torch, torch.nn as nn
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(__file__))
from multisnr import MultiSNRNoise

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from sklearn.metrics import roc_auc_score, roc_curve

CKPT_FM   = r"checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt"
CKPT_P1   = r"outputs/lora_multitask_snr_a07/lora_multitask_snr_best.pt"
CACHET    = r"data/processed/cachet"
CPSC_MC   = r"data/processed/cpsc2018_mc/test"
NSTDB     = r"data/raw/nstdb"
PTT       = r"../WidU_multimodal_fusion/raw/ptt_ppg"
LEAD = 1  # slot II = 패치 단일리드

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

def load_models(dev):
    from fairseq_signals.utils.checkpoint_utils import load_model_and_task
    res=load_model_and_task(CKPT_FM)
    bb=next(r for r in (res if isinstance(res,(list,tuple)) else [res]) if hasattr(r,'parameters')).to(dev)
    for p in bb.parameters(): p.requires_grad_(False)
    inject(bb); ck=torch.load(CKPT_P1,map_location=dev); bb.load_state_dict(ck['backbone_lora'],strict=False)
    hb=Head(1).to(dev); hb.load_state_dict(ck['head_bin_state']);
    hm=Head(5).to(dev); hm.load_state_dict(ck['head_mc_state'])
    bb.eval(); hb.eval(); hm.eval()
    return bb,hb,hm

@torch.no_grad()
def run(bb,hb,hm,x,dev,bs=32):
    es,mc=[],[]
    for i in range(0,len(x),bs):
        xb=torch.tensor(x[i:i+bs],dtype=torch.float32,device=dev)
        emb=bb(source=xb,padding_mask=None,features_only=True)['x'].mean(1)
        es.append(torch.sigmoid(hb(emb)).cpu().numpy())
        mc.append(torch.softmax(hm(emb),-1).cpu().numpy())
    return np.concatenate(es),np.concatenate(mc)

def single_lead(sig12):
    """(12,5000)에서 slot LEAD만 남기고 0-fill (패치 단일리드 시나리오)."""
    out=np.zeros_like(sig12); out[LEAD,:]=sig12[LEAD,:]; return out

def sens_spec(y,p,thr):
    yp=(p>=thr).astype(int)
    tp=((yp==1)&(y==1)).sum(); fn=((yp==0)&(y==1)).sum()
    tn=((yp==0)&(y==0)).sum(); fp=((yp==1)&(y==0)).sum()
    sens=tp/max(tp+fn,1); spec=tn/max(tn+fp,1); far=fp/max(fp+tn,1)
    return sens,spec,far

def main():
    set_det(42); dev='cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev} | 단일리드 slot{LEAD}(II) 패치 시나리오 | 결정성 고정")
    bb,hb,hm=load_models(dev)

    # ===== 척추1: CACHET =====
    print("\n"+"="*66+"\n[척추1] CACHET 패치 단일리드 AF 전이 (배포현실)\n"+"="*66)
    cs=np.load(os.path.join(CACHET,'signals.npy')); cl=np.load(os.path.join(CACHET,'labels.npy'))
    es,mc=run(bb,hb,hm,cs,dev)
    au=roc_auc_score(cl,es)
    # 운영점: 특이도 0.90 고정 시 민감도
    fpr,tpr,th=roc_curve(cl,es); spec=1-fpr
    i90=np.searchsorted(spec[::-1],0.90); sens90=tpr[::-1][i90] if i90<len(tpr) else float('nan')
    s5,sp5,far5=sens_spec(cl,es,0.5)
    print(f"  AF AUROC = {au:.4f}  (N={len(cl)}, AF={int(cl.sum())})")
    print(f"  @임계0.5 : 민감도={s5:.3f} 특이도={sp5:.3f} 오경보율={far5:.3f}")
    print(f"  @특이도0.90 운영점: 민감도={sens90:.3f}")

    # ===== 척추2: NSTDB 모션 오경보 =====
    print("\n"+"="*66+"\n[척추2] NSTDB 모션 오경보 억제 (CPSC NSR 단일리드)\n"+"="*66)
    sig=np.load(os.path.join(CPSC_MC,'signals.npy')); lab=np.load(os.path.join(CPSC_MC,'labels.npy'))
    nsr=sig[lab==0][:160]                      # NSR만 (모두 음성 → 양성=오경보)
    nsr=np.stack([single_lead(s) for s in nsr])
    aug=MultiSNRNoise(nstdb_dir=NSTDB,device=dev,seed=42)
    print(f"  NSR {len(nsr)}개, emergency 임계 0.5 기준 오경보율")
    print(f"  {'조건':>8} {'오경보율':>8}")
    for cond in ['clean',12,6,0,-6]:
        x = nsr if cond=='clean' else aug.inject_fixed(torch.tensor(nsr,dtype=torch.float32,device=dev),float(cond)).cpu().numpy()
        es,mc=run(bb,hb,hm,x,dev)
        far=(es>=0.5).mean()
        print(f"  {str(cond):>8} {far:>8.3f}")
    print("  → 오경보율이 노이즈로 오르면 모션을 ECG 비정상으로 오인 — 결합기에서 차단 필요.")

    # ===== 척추3: PTT-PPG 운동 joint =====
    print("\n"+"="*66+"\n[척추3] PTT-PPG 운동 cardiac 오발화 (실 ECG+IMU joint)\n"+"="*66)
    import wfdb
    subs=sorted(set(f.split('_')[0] for f in os.listdir(PTT) if f.endswith('.hea')))
    res_act={'sit':[], 'walk':[], 'run':[]}
    for sub in subs:
        for act in ['sit','walk','run']:
            rp=os.path.join(PTT,f"{sub}_{act}")
            if not os.path.exists(rp+'.hea'): continue
            rec=wfdb.rdrecord(rp,channels=[0])      # ECG만
            ecg=rec.p_signal[:,0].astype(np.float32)
            # 30s 이후, 10s 창 12개 (균등)
            wins=[]
            starts=np.linspace(15000, len(ecg)-5000, 12).astype(int)
            for st in starts:
                seg=np.zeros((12,5000),dtype=np.float32); seg[LEAD]=ecg[st:st+5000]
                wins.append(seg)
            es,mc=run(bb,hb,hm,np.stack(wins),dev)
            res_act[act].append(es.mean())
    print(f"  {'활동':>6} {'emergency평균':>12} {'오발화율@0.5':>12}")
    allfar={}
    for act in ['sit','walk','run']:
        arr=np.array(res_act[act]); far=(arr>=0.5).mean()
        allfar[act]=far
        print(f"  {act:>6} {arr.mean():>12.3f} {far:>12.3f}")
    print(f"  → 운동(run)에서 emergency가 sit 대비 크게 오르면 동성빈맥 오발화 위험.")
    print(f"     IMU 활동량 룰(고활동→cardiac 억제)이 결합기에서 이를 차단해야 함.")
    print("\n"+"="*66+"\n[척추 완료]\n"+"="*66)

if __name__=="__main__":
    main()
