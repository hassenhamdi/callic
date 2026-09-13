"""MGCF: Masked Gated ConvFormer (Fig.1b, MetaFormer-style).

Input -> 3×3 Type-B masked-conv embedding -> N MGCF blocks ->
1×1 Parameter Projection to discrete logistic mixture (PixelCNN++ style).

Each block: x -> LN -> MCG -> resid -> LN -> MLP -> resid,
MLP = two 1×1 (linear) layers with GELU, expansion 4×.

Paper choice (Table 3 bold): N=3 blocks (depth), dim=128, k=7.
Paper reports 575K params; with MLP×4 and K=10 mixtures this
skeleton counts 580964 (within ~1% — difference is the inferred
MLP expansion / K which the paper does not state explicitly).
Count is asserted in code/tests as 570–590K and logged exactly;
K and expansion remain configurable.
"""

import torch.nn as nn
import torch.nn.functional as F

from .mcg import MCG, causal_mask


class MaskedConv2d(nn.Module):
    def __init__(self, cin, cout, k, typ):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=k // 2)
        self.register_buffer(
            "causal_buf", causal_mask(k, typ).view(1, 1, k, k).repeat(cout, cin, 1, 1)
        )
        self.k = k

    def forward(self, x):
        buf = self.get_buffer("causal_buf")
        w = self.conv.weight * buf
        return F.conv2d(x, w, self.conv.bias, padding=self.k // 2)


class MGCFBlock(nn.Module):
    def __init__(self, dim=128, k=7):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.mcg = MCG(dim, k, "A")
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1), nn.GELU(), nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        h = x.permute(0, 2, 3, 1)
        h = self.ln1(h).permute(0, 3, 1, 2)
        x = x + self.mcg(h)
        h = x.permute(0, 2, 3, 1)
        h = self.ln2(h).permute(0, 3, 1, 2)
        return x + self.mlp(h)


class MGCF(nn.Module):
    def __init__(self, dim=128, depth=3, k=7, mixtures=10):
        super().__init__()
        self.embed = MaskedConv2d(3, dim, 3, "B")
        self.blocks = nn.ModuleList([MGCFBlock(dim, k) for _ in range(depth)])
        self.head = nn.Conv2d(dim, mixtures * 10, 1)

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
