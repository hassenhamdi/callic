"""MCG: Masked Convolutional Gating (Eq.5, Fig.1a).

Eq.5:
  A_M = DWConv_{k×k}(W_A X, M)
  V   = W_V X
  MCG(X) = swish(A_M) ⊙ V
followed by 1×1 out-proj (Fig.1a top/bottom branches).

M is the convolutional causal mask restricting to decoded scope.
Type-B (embedding): masks current positions. Type-A (MCG blocks):
includes current + past.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(k, typ="A"):
    """Causal mask k×k. Type-A includes center, Type-B excludes it.

    Allows only rows above center, and columns left of center on the
    center row — i.e. raster-scan past. Future (bottom rows / right
    of center) is zeroed.
    """
    m = torch.zeros(k, k)
    c = k // 2
    for y in range(k):
        for x in range(k):
            if y < c or (y == c and x < c) or (y == c and x == c and typ == "A"):
                m[y, x] = 1.0
    return m


class MCG(nn.Module):
    """Masked Convolutional Gating Eq.5: swish(DWConv(W_A X)) * (W_V X)."""

    def __init__(self, dim=128, k=7, typ="A"):
        super().__init__()
        self.wa = nn.Conv2d(dim, dim, 1)
        self.wv = nn.Conv2d(dim, dim, 1)
        self.dw = nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.k = k
        self.typ = typ
        self.register_buffer("causal_buf", causal_mask(k, typ).view(1, 1, k, k))

    def masked_dw_weight(self):
        return self.dw.weight * self.get_buffer("causal_buf")

    def forward(self, x):
        buf = self.get_buffer("causal_buf")
        w = self.dw.weight * buf
        am = F.conv2d(
            self.wa(x), w, self.dw.bias, padding=self.k // 2, groups=w.shape[0]
        )
        v = self.wv(x)
        return self.proj(F.silu(am) * v)
