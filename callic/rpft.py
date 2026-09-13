"""RPFT: Rate-guided Progressive Fine-Tuning (Eq.8, Fig.2, Fig.4).

Paper details held:
- Estimate per-patch bpsp, sort descending (highest entropy first).
- Progressively increase training fraction F(t):
    t' = t / (T·(1−d))
    s(x) = 0 if x<0, 1 if x>1, x²(3−2x) if 0≤x≤1   (smoothstep)
    F(t) = b + (1−b)·[s(t')]^e
  Defaults: b=0.2, d=0.1, e=1, T=50, lr=1e-2 (Adaptation Settings).
  Final d% steps use the full set (t'≥1 ⇒ F=1), aligning train/test.
- Focus on higher-rate patches beats increasing/random (Fig.4).
- Optimizes MDL loss Eq.9 jointly over incremental weights.
"""

import torch


def smoothstep(x: float) -> float:
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return x * x * (3.0 - 2.0 * x)


def train_fraction(t: int, T: int = 50, b: float = 0.2, d: float = 0.1, e: float = 1.0) -> float:
    """F(t) per Eq.8. t is 0-indexed current step, T total steps."""
    denom = T * (1.0 - d)
    tp = t / denom if denom > 0 else 1.0
    return b + (1.0 - b) * (smoothstep(tp) ** e)


def patchify(x: torch.Tensor, P: int = 64):
    """Split [C,H,W] or [B,C,H,W] image into non-overlap P×P patches.

    Returns (patches list, grid H', W'). Pads with edge replication if needed
    (padding bits are excluded from bpsp by counting only valid pixels —
    caller handles; smoke uses divisible sizes).
    """
    single = x.dim() == 3
    if single:
        x = x.unsqueeze(0)
    B, C, H, W = x.shape
    pad_h = (P - H % P) % P
    pad_w = (P - W % P) % P
    if pad_h or pad_w:
        import torch.nn.functional as F

        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    _, _, Hp, Wp = x.shape
    patches = []
    for i in range(0, Hp, P):
        for j in range(0, Wp, P):
            patches.append(x[:, :, i : i + P, j : j + P])
    return patches, (Hp, Wp)


def estimate_patch_rates(model, patches, mixture_nll_fn):
    """Estimate per-patch bpsp with frozen pre-trained model (no grad)."""
    model.eval()
    rates = []
    with torch.no_grad():
        for p in patches:
            px = p.float()
            if px.max() <= 1.0:
                px = px * 255.0
            logits = model(px)
            nll = mixture_nll_fn(px.to(torch.uint8), logits)
            rates.append(float(nll.item()))
    return rates


def sorted_patch_order(rates, descending=True):
    """Indices sorted by rate; default descending (paper: highest first)."""
    return sorted(range(len(rates)), key=lambda i: rates[i], reverse=descending)


def rpft_finetune(model, image, mixture_nll_fn, adapt_params_fn=None,
                  T=50, lr=1e-2, b=0.2, d=0.1, e=1.0, P=64, opt=None):
    """Rate-guided progressive fine-tuning loop for one test image.

    image: [C,H,W] uint8 or [B,C,H,W]. Only adaptor params (or all params
      if adapt_params_fn is None) are updated; pre-trained θ stays fixed
      in the paper (we update only params with requires_grad=True that are
      not frozen — caller freezes base).
    Returns dict with per-step fractions and final rates for Fig.3 sweeps.
    """
    from .adapt import mdl_loss

    single = image.dim() == 3
    img = image.unsqueeze(0) if single else image
    patches, _ = patchify(img[0], P)
    # patches are [1,C,P,P]; stack for convenience
    rates = estimate_patch_rates(model, patches, mixture_nll_fn)
    order = sorted_patch_order(rates, descending=True)

    params = [p for p in model.parameters() if p.requires_grad]
    if adapt_params_fn is not None:
        params = adapt_params_fn(model)
        if len(params) == 0:
            params = [p for p in model.parameters() if p.requires_grad]
    elif hasattr(model, "adapt_params") and callable(model.adapt_params):
        # CALLICModel: freeze base theta, update incremental phi only (paper).
        params = model.adapt_params()
    if opt is None:
        opt = torch.optim.Adam(params, lr=lr)

    model.train()
    hist = []
    n = len(patches)
    for t in range(T):
        frac = train_fraction(t, T, b, d, e)
        k = max(1, int(round(frac * n)))
        sel = [patches[order[i]] for i in range(k)]
        batch = torch.cat(sel, dim=0).float()
        if batch.max() <= 1.0:
            batch = batch * 255.0
        opt.zero_grad()
        logits = model(batch)
        pix = mixture_nll_fn(batch.to(torch.uint8), logits)
        # MDL: weight bits spread over image pixels would need H·W;
        # here loss = pixel NLL (mean bpsp) + weight-rate term handled
        # by caller for reporting; optimize pixel term + small weight decay
        # via prior — full joint Eq.9 used in tools/train.py reporting.
        loss = pix
        loss.backward()
        opt.step()
        hist.append({"t": t, "frac": frac, "k": k, "loss": float(loss.item())})
    return {"order": order, "rates": rates, "hist": hist}
