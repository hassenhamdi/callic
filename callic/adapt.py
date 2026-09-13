"""LoRA (Eq.6) + Tucker DWConv (Eq.7) + STE quant + MDL loss (Eq.9).

Paper details held:
- LoRA Eq.6 on W_A, W_V and MLP W_up: W' = W + A B, A in R^{m×r}, B in R^{r×n}.
- Tucker Eq.7 on masked DWConv W_mc in R^{m×1×k×k}:
    ΔW = I ×1 A ×3 C ×4 D,  W'_mc = M ⊙ (W_mc + ΔW),
  where I in R^{r1×1×r2×r3} is the (learnable, zero-init) core,
  A in R^{m×r1}, C in R^{k×r2}, D in R^{k×r3}, ×n is mode-n product,
  M is the causal mask. Merged with zero infer overhead.
- Adapted: WA/WV + first linear Wup in MLP + DWConv in every MCG block.
- Rank config targeting ~25K mergeable (paper: CALLIC adds 25K):
    WA/WV r=8, Wup r=4, DWConv (r1=8, r2=4, r3=4), depth=3, dim=128, k=7.
  Exact count logged; tolerance 23–27K asserted in tests.
- Quant: step w=0.05 (<1, paper default). STE for inference
    φ̂ = sg(⌊φ/w⌉·w − φ) + φ; uniform noise φ̃ = φ + U(−w/2, w/2) for rate.
- Prior: static logistic zero-mean scale s=0.05 on φ̃.
- Loss Eq.9: L = −log p_s(φ̃) + Σ_i −log q(x_Gi | x_G<i; θ, φ̂).
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """LoRA wrapper for 1×1 conv (= linear over channels), Eq.6."""

    def __init__(self, base: nn.Conv2d, r=8):
        super().__init__()
        cin, cout = base.in_channels, base.out_channels
        assert base.kernel_size == (1, 1), "LoRA only wraps 1×1 convs (W_A/W_V/W_up)"
        self.base = base
        self.A = nn.Parameter(torch.zeros(cout, r))
        self.B = nn.Parameter(torch.zeros(r, cin))
        nn.init.normal_(self.A, std=0.02)
        nn.init.zeros_(self.B)
        self.r = r

    def delta(self):
        return self.A @ self.B  # [cout, cin]

    def merged_weight(self):
        w = self.base.weight.squeeze(-1).squeeze(-1)  # [cout,cin] for 1x1
        return (w + self.delta()).unsqueeze(-1).unsqueeze(-1)

    def forward(self, x):
        import torch.nn.functional as F

        return F.conv2d(x, self.merged_weight(), self.base.bias)


class TuckerDWConvAdapt(nn.Module):
    """Tucker low-rank adaptor for masked depth-wise conv, Eq.7.

    base: nn.Conv2d groups=dim, weight [m,1,k,k].
    ΔW[m,0,j,l] = Σ_{a,b,c} core[a,0,b,c] · A[m,a] · C[j,b] · D[l,c].
    Forward uses M ⊙ (W + ΔW) with the block's causal mask M.
    All adaptor params init 0 so ΔW=0 at start (identity adaptation).
    """

    def __init__(self, base: nn.Conv2d, mask: torch.Tensor, r1=8, r2=4, r3=4):
        super().__init__()
        assert base.groups == base.in_channels == base.out_channels
        m, _, k, k2 = base.weight.shape
        assert k == k2
        self.base = base
        # mask: [1,1,k,k] broadcastable — stored as buffer reference
        self.register_buffer("mask", mask.clone())
        self.A = nn.Parameter(torch.zeros(m, r1))
        self.C = nn.Parameter(torch.zeros(k, r2))
        self.D = nn.Parameter(torch.zeros(k, r3))
        self.core = nn.Parameter(torch.zeros(r1, 1, r2, r3))
        nn.init.normal_(self.A, std=0.02)
        nn.init.normal_(self.C, std=0.02)
        nn.init.normal_(self.D, std=0.02)
        nn.init.zeros_(self.core)
        self.r1, self.r2, self.r3 = r1, r2, r3

    def delta(self):
        # einsum: core[a,0,b,c] * A[m,a] * C[j,b] * D[l,c] -> [m,1,j,l]
        # core squeeze dim1: [r1,r2,r3]
        g = self.core.squeeze(1)  # [r1,r2,r3]
        # Δ[m,j,l] = Σ_abc g[a,b,c] A[m,a] C[j,b] D[l,c]
        d = torch.einsum("abc,ma,jb,lc->mjl", g, self.A, self.C, self.D)
        return d.unsqueeze(1)  # [m,1,k,k]

    def merged_weight(self):
        return self.mask * (self.base.weight + self.delta())

    def forward(self, x):
        import torch.nn.functional as F

        w = self.merged_weight()
        return F.conv2d(
            x, w, self.base.bias, padding=self.base.kernel_size[0] // 2,
            groups=w.shape[0],
        )


def ste_quant(phi: torch.Tensor, w: float = 0.05):
    """STE quant for inference: φ̂ = sg(round(φ/w)·w − φ) + φ."""
    q = torch.round(phi / w) * w
    return (q - phi).detach() + phi


def noisy_weights_for_rate(phi: torch.Tensor, w: float = 0.05):
    """Uniform-noise surrogate φ̃ = φ + U(−w/2, w/2) for rate estimate."""
    return phi + (torch.rand_like(phi) - 0.5) * w


def logistic_logpdf_zero_mean(x: torch.Tensor, s: float = 0.05):
    """Log-pdf of Logistic(0, s): −x/s − log s − 2·softplus(−x/s)."""
    return -x / s - math.log(s) - 2.0 * torch.nn.functional.softplus(-x / s)


def incremental_rate_bits(params, s: float = 0.05, w: float = 0.05):
    """Rate for incremental weights: Σ −log p_s(φ̃) in bits.

    Uses uniform-noise surrogate φ̃. For quantized weights with step w,
    probability mass ≈ pdf(φ̃)·w, so −log mass = −log pdf − log w
    (both in nats → bits). The −log w term (w=0.05 ⇒ +4.32 bits/param)
    keeps discrete rates positive; included explicitly.
    """
    total_nats = torch.zeros((), device=params[0].device)
    for p in params:
        noisy = noisy_weights_for_rate(p, w)
        total_nats = total_nats + (-logistic_logpdf_zero_mean(noisy, s)).sum()
        total_nats = total_nats + (-math.log(w)) * p.numel()
    return total_nats / math.log(2.0)


def mdl_loss(pixel_nll_bits: torch.Tensor, adapt_params, s=0.05, w=0.05):
    """Eq.9: L = −log p_s(φ̃) + Σ_i −log q(x_Gi | x_G<i; θ, φ̂).

    pixel_nll_bits: mean or summed pixel NLL in bits (from mixture head
      evaluated with STE-quantized φ̂ merged weights).
    Returns total loss in bits (weight bits + pixel bits).
    """
    wbits = incremental_rate_bits(adapt_params, s, w) if len(adapt_params) else torch.zeros(())
    # pixel_nll_bits is mean bpsp-style; caller scales to sum if needed.
    return wbits + pixel_nll_bits


def collect_adapt_params(mgcf, prefix_filter=None):
    """Collect incremental (A,B,core,C,D) params for rate computation."""
    out = []
    for n, p in mgcf.named_parameters():
        if any(k in n for k in ("lora_A", "lora_B", "tucker", ".A", ".B", ".C", ".D", ".core")):
            out.append(p)
        elif prefix_filter and prefix_filter in n:
            out.append(p)
    return out


def count_mergeable(mgcf=None, r_wa=8, r_wv=8, r_up=4, r1=8, r2=4, r3=4,
                    dim=128, depth=3, k=7):
    """Exact mergeable count for the paper's rank config.

    Per block: LoRA WA (dim·r_wa + r_wa·dim) + LoRA WV same +
      LoRA Wup (dim·r_up + r_up·dim·4) +
      Tucker DWConv (dim·r1 + k·r2 + k·r3 + r1·r2·r3 core).
    × depth. Defaults: 3 blocks → 23592 (within 23–27K).
    mgcf arg accepted for API compat; count is config-derived.
    """
    per_block = (dim * r_wa + r_wa * dim) + (dim * r_wv + r_wv * dim)
    per_block += dim * r_up + r_up * dim * 4
    per_block += dim * r1 + k * r2 + k * r3 + r1 * r2 * r3
    return per_block * depth


class CALLICModel(nn.Module):
    """Content-adaptive wrapper: frozen MGCF base + mergeable LoRA/Tucker deltas.

    Wires Eq.6 (LoRA on WA/WV/Wup per block) + Eq.7 (Tucker on DWConv per
    block) onto a frozen MGCF without mutating it until merge. All deltas
    zero-init so adapted(x) == base(x) at start. STE-quantized merged
    weights used for inference (phi-hat); uniform-noise surrogates for rate.
    merge_() writes quantized merged weights back into base for zero-overhead
    inference per paper.
    """

    def __init__(self, mgcf, r_wa=8, r_wv=8, r_up=4, r1=8, r2=4, r3=4):
        super().__init__()
        self.base = mgcf
        for p in self.base.parameters():
            p.requires_grad_(False)
        import torch.nn.functional as F  # noqa: F401 (kept for forward clarity)

        self.blocks = nn.ModuleList()
        for b in mgcf.blocks:
            dim = b.mcg.wa.in_channels
            k = b.mcg.dw.kernel_size[0]
            up_dim = b.mlp[0].out_channels
            blk = nn.ParameterDict({
                "lora_wa_A": nn.Parameter(torch.zeros(dim, r_wa)),
                "lora_wa_B": nn.Parameter(torch.zeros(r_wa, dim)),
                "lora_wv_A": nn.Parameter(torch.zeros(dim, r_wv)),
                "lora_wv_B": nn.Parameter(torch.zeros(r_wv, dim)),
                "lora_up_A": nn.Parameter(torch.zeros(dim, r_up)),
                "lora_up_B": nn.Parameter(torch.zeros(r_up, up_dim)),
                "tucker_A": nn.Parameter(torch.zeros(dim, r1)),
                "tucker_C": nn.Parameter(torch.zeros(k, r2)),
                "tucker_D": nn.Parameter(torch.zeros(k, r3)),
                "tucker_core": nn.Parameter(torch.zeros(r1, 1, r2, r3)),
            })
            self.blocks.append(blk)
        for blk in self.blocks:
            for n in ("lora_wa_A", "lora_wv_A", "lora_up_A", "tucker_A",
                      "tucker_C", "tucker_D"):
                nn.init.normal_(blk[n], std=0.02)
        # Keep adaptor params on the base model's device (CPU/GPU).
        self.to(next(mgcf.parameters()).device)
        self.r = {"r_wa": r_wa, "r_wv": r_wv, "r_up": r_up,
                  "r1": r1, "r2": r2, "r3": r3}

    def adapt_params(self):
        return [p for blk in self.blocks for p in blk.parameters()]

    def _merged_1x1(self, w4, A, B, w_step=0.05, quantize=False, transpose=False):
        import torch.nn.functional as F  # noqa: F401

        d = A @ B  # [cout, cin] (or [cin, cout] if transpose)
        if transpose:
            d = d.t()
        if quantize:
            d = ste_quant(d, w_step)
        return w4 + d.unsqueeze(-1).unsqueeze(-1)

    def _merged_dw(self, w, mask, blk, w_step=0.05, quantize=False):
        g = blk["tucker_core"].squeeze(1)
        d = torch.einsum("abc,ma,jb,lc->mjl", g, blk["tucker_A"],
                         blk["tucker_C"], blk["tucker_D"]).unsqueeze(1)
        if quantize:
            d = ste_quant(d, w_step)
        return mask * (w + d)

    def forward(self, x, quantize=False, w_step=0.05):
        import torch.nn.functional as F

        h = self.base.embed(x)
        for bi, b in enumerate(self.base.blocks):
            blk = self.blocks[bi]
            # LN -> MCG(resid)
            hh = h.permute(0, 2, 3, 1)
            hh = b.ln1(hh).permute(0, 3, 1, 2)
            wa = self._merged_1x1(b.mcg.wa.weight,
                                  blk["lora_wa_A"], blk["lora_wa_B"], w_step, quantize)
            wv = self._merged_1x1(b.mcg.wv.weight,
                                  blk["lora_wv_A"], blk["lora_wv_B"], w_step, quantize)
            ax = F.conv2d(hh, wa, b.mcg.wa.bias)
            vx = F.conv2d(hh, wv, b.mcg.wv.bias)
            mask = b.mcg.get_buffer("causal_buf")
            dw = self._merged_dw(b.mcg.dw.weight, mask, blk, w_step, quantize)
            am = F.conv2d(ax, dw, b.mcg.dw.bias,
                          padding=b.mcg.k // 2, groups=dw.shape[0])
            gate = b.mcg.proj(F.silu(am) * vx)
            h = h + gate
            # LN -> MLP(resid) with merged Wup ([dim,up_dim] delta transposed)
            hh = h.permute(0, 2, 3, 1)
            hh = b.ln2(hh).permute(0, 3, 1, 2)
            wup = self._merged_1x1(
                b.mlp[0].weight,
                blk["lora_up_A"], blk["lora_up_B"], w_step, quantize,
                transpose=True)
            h1 = F.conv2d(hh, wup, b.mlp[0].bias)
            h1 = F.gelu(h1)
            h = h + F.conv2d(h1, b.mlp[2].weight, b.mlp[2].bias)
        return self.base.head(h)

    def rate_bits(self, s=0.05, w=0.05):
        return incremental_rate_bits(self.adapt_params(), s=s, w=w)

    @torch.no_grad()
    def merge_(self, w_step=0.05):
        """Write STE-quantized merged weights into base (zero infer overhead)."""
        for bi, b in enumerate(self.base.blocks):
            blk = self.blocks[bi]
            dwa = self._merged_1x1(b.mcg.wa.weight,
                                   blk["lora_wa_A"], blk["lora_wa_B"], w_step, True)
            b.mcg.wa.weight.copy_(dwa)
            dwv = self._merged_1x1(b.mcg.wv.weight,
                                   blk["lora_wv_A"], blk["lora_wv_B"], w_step, True)
            b.mcg.wv.weight.copy_(dwv)
            mask = b.mcg.get_buffer("causal_buf")
            b.mcg.dw.weight.copy_(self._merged_dw(b.mcg.dw.weight, mask, blk, w_step, True))
            wup = self._merged_1x1(b.mlp[0].weight,
                                   blk["lora_up_A"], blk["lora_up_B"], w_step, True,
                                   transpose=True)
            b.mlp[0].weight.copy_(wup)
        for p in self.base.parameters():
            p.requires_grad_(False)
        return self.base
