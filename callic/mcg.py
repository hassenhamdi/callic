import torch
import torch.nn as nn


class MCG(nn.Module):
    """Masked Convolutional Gating Eq.5: swish(DWConv(W_A X)) * (W_V X)."""
    def __init__(self, dim=128, k=7):
        super().__init__()
        self.wa = nn.Conv2d(dim, dim, 1)
        self.wv = nn.Conv2d(dim, dim, 1)
        self.dw = nn.Conv2d(dim, dim, k, padding=k//2, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.k = k
    def forward(self, x, mask=None):
        am = self.dw(self.wa(x))
        if mask is not None:
            am = am * mask
        v = self.wv(x)
        return self.proj(torch.nn.functional.silu(am) * v)
