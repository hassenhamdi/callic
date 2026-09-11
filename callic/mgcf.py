import torch.nn as nn

from .mcg import MCG


class MGCFBlock(nn.Module):
    def __init__(self, dim=128, k=7):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.mcg = MCG(dim, k)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Conv2d(dim, dim*4, 1), nn.GELU(), nn.Conv2d(dim*4, dim, 1))
    def forward(self, x, mask=None):
        h = x.permute(0,2,3,1); h = self.ln1(h).permute(0,3,1,2)
        x = x + self.mcg(h, mask)
        h = x.permute(0,2,3,1); h = self.ln2(h).permute(0,3,1,2)
        return x + self.mlp(h)
class MGCF(nn.Module):
    def __init__(self, dim=128, depth=3, k=7, mixtures=10):
        super().__init__()
        self.embed = nn.Conv2d(3, dim, 3, padding=1)
        self.blocks = nn.ModuleList([MGCFBlock(dim, k) for _ in range(depth)])
        self.head = nn.Conv2d(dim, mixtures*10, 1)
    def forward(self, x, mask=None):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h, mask)
        return self.head(h)
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
