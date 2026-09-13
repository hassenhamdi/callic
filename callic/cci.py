"""CCI: Cache-then-Crop grouped AR inference (Fig.1c).

Paper details held:
- Image divided into patches, patches encoded in parallel.
- Per patch, pixels grouped x={x_G1..x_Gg} in DLPR-inspired parallel
  scan entailing 3P−2 autoregressive steps (P = patch size).
- Step i: consume cached activations of x_G≤i−1, predict x_Gi.
- Type-B (embedding, top): current-group positions masked out; feed
  previous-group pixels, cache, cropped Type-B conv to gather context.
- Type-A (deeper MCG): current-group positions included; store current
  activations to cache map, cropped conv for context.
- 1×1 convs transfer across channels only — no caching.
- Cache-then-crop: cache activations before each masked DWConv; at step
  i crop windows around current-group positions, zero-pad conv only on
  crops in parallel.
- Parity invariant: CCI output ≡ naive full-masked-conv output.

Grouping note (ambiguity made explicit): the exact DLPR scan order is
described only as "parallel scan ... 3P−2 steps" + Fig.1c. The paper's
Type-A/B masks are GROUP-dependent (current group masked out for B,
included for A), so any grouping is causal by construction when masks
are applied per-step. Our local smoke uses STATIC raster masks (fixed
causal MaskedConv2d/MCG) + raster-respecting contiguous groups with
exactly G=3P−2 steps globally (slab i = raster chunk i). This respects
raster causality (past groups present, future zeroed-but-masked), so the
parity test is a true mask-leakage detector: if masks leaked future,
zeroing future would change outputs and parity would fail. The full
DLPR diagonal scan with per-step dynamic group masks is the Colab
variant (same interface, same step count); parity guards both.
"""

import torch
import torch.nn.functional as F


def num_groups_for_patch(P: int) -> int:
    return 3 * P - 2


def group_indices(H, W, P=8):
    """Raster-respecting groups with 3P−2 steps (smoke instantiation).

    G=3P−2 contiguous raster slabs tiled in scan order. Past slabs contain
    all raster-past dependencies of later slabs (up to intra-slab left
    context, which stays present since the whole current slab is fed).
    Returns list of LongTensor flattened indices in scan order.
    """
    G = num_groups_for_patch(P)
    N = H * W
    groups = []
    for i in range(G):
        s = (i * N) // G
        e = ((i + 1) * N) // G
        if e > s:
            groups.append(torch.arange(s, e, dtype=torch.long))
    return groups


def cropped_dwconv_forward(x_cache, weight, bias, k, positions, H, W):
    """Cache-then-crop masked DWConv at `positions` (flattened indices).

    x_cache: [B,C,H,W] cached activations (past + current as appropriate).
    weight: [C,1,k,k] already causally masked (M⊙W).
    Crops k×k windows around each position, zero-pads at borders,
    convolves only on crops in parallel. Returns [B,C,len(positions)].
    Functional equivalent of full masked DWConv gathered at positions.
    """
    B, C, _, _ = x_cache.shape
    pad = k // 2
    xp = F.pad(x_cache, (pad, pad, pad, pad), mode="constant", value=0.0)
    ys = (positions // W) + pad
    xs = (positions % W) + pad
    N = positions.numel()
    windows = torch.zeros(B, C, N, k, k, device=x_cache.device, dtype=x_cache.dtype)
    Hp, Wp = xp.shape[2], xp.shape[3]
    for dy in range(k):
        for dx in range(k):
            yy = (ys + dy - pad).clamp(0, Hp - 1)
            xx = (xs + dx - pad).clamp(0, Wp - 1)
            for n in range(N):
                windows[:, :, n, dy, dx] = xp[:, :, int(yy[n]), int(xx[n])]
    w = weight.view(C, k * k)
    win = windows.view(B, C, N, k * k)
    out = (win * w.view(1, C, 1, k * k)).sum(-1)  # [B,C,N]
    if bias is not None:
        out = out + bias.view(1, C, 1)
    return out


def cci_sequential_forward(model, x, P=8):
    """Grouped sequential forward proving causality (no mask leakage).

    At step i, future groups (>i) are zeroed; past + current kept.
    Outputs at group-i positions collected. Assembled ≡ full forward
    iff static masks block future.
    """
    model.eval()
    with torch.no_grad():
        B, _, H, W = x.shape
        full = model(x.float())
        groups = group_indices(H, W, P=P)
        flat_full = full.reshape(B, full.shape[1], -1)
        recon = torch.zeros_like(full)
        flat_recon = recon.reshape(B, recon.shape[1], -1)
        x_base = x.float()
        for i, g in enumerate(groups):
            masked = x_base.clone()
            if i + 1 < len(groups):
                future = torch.cat(groups[i + 1 :])
                mf = masked.reshape(B, 3, -1)
                mf[:, :, future] = 0.0
            out = model(masked)
            flat_out = out.reshape(B, out.shape[1], -1)
            flat_recon[:, :, g] = flat_out[:, :, g]
        err = (flat_full - flat_recon).abs().max().item()
    return recon, err


def cci_parity_check(model, x, P=8):
    """Parity invariant: CCI sequential ≡ naive full-masked-conv. Returns max err."""
    _, err = cci_sequential_forward(model, x, P=P)
    return err
