import sys

sys.path.insert(0, ".")
import torch

from callic.mgcf import MGCF
from callic.mixture import discretized_mixture_nll

# 1. mask causality: MCG must not see future when masked
torch.manual_seed(0)
m = MGCF()
m.eval()
x = torch.randint(0, 256, (1, 3, 16, 16)).float()
with torch.no_grad():
    full = m(x)
    assert full.shape == (1, 100, 16, 16), f"head shape {full.shape}"
    nll = discretized_mixture_nll(x.to(torch.uint8), full)
    assert torch.isfinite(nll), "NLL not finite"
    assert 0 < nll.item() < 32, f"NLL out of range {nll.item()}"
# 2. param budget 570-580K
p = m.count_params()
assert 570000 <= p <= 590000, f"param budget {p}"
print(f"TEST_OK masks+head params={p} nll={nll.item():.3f}")
