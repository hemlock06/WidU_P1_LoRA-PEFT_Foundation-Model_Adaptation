"""
Verify the LoRA explicit-projection attention path equals the original fused
F.multi_head_attention_forward path when the LoRA delta is zero.

If this passes, enabling LoRA does not perturb the pretrained model at step 0
(standard LoRA init: B=0 -> zero delta), so baseline behavior is preserved.
"""
import torch
from fairseq_signals.modules.multi_head_attention import MultiHeadAttention

torch.manual_seed(0)

embed_dim, n_heads = 768, 12
attn = MultiHeadAttention(embed_dim, n_heads, dropout=0.0, self_attention=True)
attn.eval()

T, B = 50, 4
x = torch.randn(T, B, embed_dim)

# Original fused path
with torch.no_grad():
    out_fused, _ = attn(query=x, key=x, value=x, need_weights=False)

# Enable LoRA on all projections; B=0 so delta is zero -> must match fused path
attn.enable_lora(r=8, alpha=16, dropout=0.0, targets={"q", "k", "v", "out"})
attn.eval()
with torch.no_grad():
    out_lora, _ = attn(query=x, key=x, value=x, need_weights=False)

max_abs = (out_fused - out_lora).abs().max().item()
print(f"max abs diff (zero-delta LoRA vs fused): {max_abs:.3e}")
assert max_abs < 1e-5, f"MISMATCH: {max_abs}"
print("PASS: zero-delta LoRA path matches fused path")

# Now perturb lora_B so the delta is non-zero -> output MUST change
with torch.no_grad():
    attn.lora_B_q.normal_(0, 0.02)
    out_perturbed, _ = attn(query=x, key=x, value=x, need_weights=False)
delta = (out_perturbed - out_fused).abs().max().item()
print(f"max abs diff (non-zero LoRA delta): {delta:.3e}")
assert delta > 1e-4, "LoRA delta had no effect — injection is not wired correctly"
print("PASS: non-zero LoRA delta changes the output (LoRA path is live)")

# Gradient check: only lora_* params should receive grad after freezing logic.
# Here just confirm lora params are leaf tensors with grad capability.
assert attn.lora_A_q.requires_grad and attn.lora_B_q.requires_grad
print("PASS: LoRA params are trainable")
