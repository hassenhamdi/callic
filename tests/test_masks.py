import sys

sys.path.insert(0, ".")
import torch

from callic.adapt import count_mergeable
from callic.cci import cci_parity_check
from callic.mcg import causal_mask
from callic.mgcf import MGCF
from callic.mixture import discretized_mixture_nll

torch.manual_seed(0)
# 1. causal masks: A center=1, B center=0
ma = causal_mask(7, "A")
mb = causal_mask(7, "B")
assert ma[3, 3] == 1.0 and mb[3, 3] == 0.0, "Type-A/B center wrong"
assert ma.sum() > mb.sum(), "A must include more than B"
# 2. head + NLL finite
m = MGCF()
m.eval()
x = torch.randint(0, 256, (1, 3, 16, 16)).float()
with torch.no_grad():
    full = m(x)
    assert full.shape == (1, 100, 16, 16), f"head shape {full.shape}"
    nll = discretized_mixture_nll(x.to(torch.uint8), full)
    assert torch.isfinite(nll) and 0 < nll.item() < 32
    err = cci_parity_check(m, x.to(torch.uint8)[:1])
    assert err < 1e-5, f"CCI parity {err}"
# 3. budgets
p = m.count_params()
assert 570000 <= p <= 590000, f"param budget {p}"
mp = count_mergeable(m)
assert 23000 <= mp <= 27000, f"mergeable budget {mp}"
print(
    f"TEST_OK masks A/B CCIerr={err:.1e} params={p} mergeable={mp} nll={nll.item():.3f}"
)
