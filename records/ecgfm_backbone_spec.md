# ECG-FM 백본 구조 실측 (포폴 아키텍처 도식 검증)

> 방법: `mimic_iv_ecg_physionet_pretrained.pt` 로드 → `cfg.model` + 실제 `named_modules()` 실측.
> 로드/추론만(학습 없음). 재현: `scripts/_inspect_ecgfm.py`. 모델: `wav2vec2_cmsc` (90.9M, frozen).
> ★ 주의: 체크포인트 옆 `.yaml` 사이드카는 stale(1024/24-layer로 오기)이며, **실제는 ckpt `cfg`·텐서 기준 768/12.**

## 1. cfg.model 실측값

| 항목 | 실측값 | 의미 |
|---|---|---|
| _name | `wav2vec2_cmsc` | Wav2Vec2 + CMSC 사전학습 |
| **layer_norm_first** | **False** | **→ post-norm (Post-LN)** ★ |
| encoder_layers | 12 | Transformer 층 수 |
| encoder_embed_dim | 768 | 토큰 차원 (= 출력 임베딩) |
| encoder_attention_heads | 12 | 헤드 수 (head dim 64) |
| encoder_ffn_embed_dim | 3072 | FFN 내부 (4× 확장) |
| activation_fn | None → **GELU**(기본) | FFN 활성 |
| conv_feature_layers | **`[(256, 2, 2)] × 4`** | CNN 4블록: out=256·kernel=2·stride=2 |
| conv_pos / conv_pos_groups | 128 / 16 | **conv 기반 positional** |
| in_d | 12 | 입력 = 12-lead |
| normalize | False | raw mV 입력(인스턴스 정규화 없음) |
| final_dim | 256 | (사전학습 contrastive proj용; 인코더 출력 아님) |
| dropout | 0.1 | |

## 2. 실제 모듈 (named_modules 실측)

**feature_extractor — Conv1d 4블록** (전부 bias=False):
```
conv_layers.0.0: Conv1d(in=12,  out=256, k=2, s=2)
conv_layers.1.0: Conv1d(in=256, out=256, k=2, s=2)
conv_layers.2.0: Conv1d(in=256, out=256, k=2, s=2)
conv_layers.3.0: Conv1d(in=256, out=256, k=2, s=2)
```
→ 다운샘플 2⁴ = **16×**. 입력 5000샘플(10초@500Hz) → **312 시간토큰**.

**feature projection / positional:**
```
post_extract_proj : Linear(256 → 768)          # CNN 출력 256 → 인코더 768
conv_pos.pos_conv.0: Conv1d(768→768, k=128, groups=16)   # convolutional positional embedding
```

**encoder.layers[0] = TransformerEncoderLayer:**
```
self_attn            : MultiHeadAttention (q/k/v/out_proj 각 Linear 768→768)
self_attn_layer_norm : LayerNorm(768)
fc1                  : Linear(768 → 3072)
fc2                  : Linear(3072 → 768)
final_layer_norm     : LayerNorm(768)
```
norm 판정: `cfg.layer_norm_first=False` + forward 내 분기 존재 → **Post-LN**(norm을 residual add **이후** 적용).

**LoRA 부착 (`inject_lora` 실측):** `target_suffixes = ("self_attn.q_proj", "self_attn.v_proj")`
→ **q_proj·v_proj만** 감쌈 (k_proj·out_proj·FFN 미부착). rank=8, α=16.

## 3. 도식 vs 실측 — 일치/불일치

| 도식 항목 | 실측 | 판정 |
|---|---|---|
| Transformer norm 방식 | **post-norm** (layer_norm_first=False) | 도식이 pre-norm이면 **수정 필요** |
| **feature projection** | **256 → 768** | 도식 "512→768"이면 **★수정 필요**(conv 출력은 256) |
| CNN feature extractor | **4블록 (256ch, k2/s2), 16×** | 표준 wav2vec2(7층/512ch/320×)로 그렸으면 **수정 필요** |
| positional embedding | **convolutional**(Conv1d k=128, g=16) | 사인파/학습형 absolute로 그렸으면 **수정 필요** |
| 인코더 차원·층·헤드 | 768 / 12층 / 12헤드 / FFN 3072 | 768/12/12면 일치 (1024/24는 yaml stale, 오류) |
| 입력 | 12-lead raw(정규화 없음), 첫 conv 12→256 | 일치 확인 |
| LoRA 대상 | q_proj·v_proj only | q/v면 일치 |

## 4. 결론

**도식 수정 필요 가능 항목(우선순위):**
1. **★ feature projection: 512→768 → 256→768** (ECG-FM CMSC는 256-ch conv. 표준 wav2vec2의 512 아님).
2. **★ CNN: 4블록·256ch·k2/s2·16× 다운샘플** (표준 7층/512/320× 아님) → 312 토큰.
3. **norm = post-norm(Post-LN)** 명시 (layer_norm_first=False).
4. **positional = convolutional**(Conv1d), 사인파/absolute 아님.

768-dim·12층·12헤드·FFN3072·q/v-LoRA·raw 12-lead 입력은 그대로 사용 가능. 1024/24로 적힌 게 있으면 그건 yaml stale 오류이므로 **768/12로 정정.**
