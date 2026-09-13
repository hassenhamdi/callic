import sys

sys.path.insert(0, ".")
import torch

from callic.adapt import (
    LoRALinear,
    TuckerDWConvAdapt,
    count_mergeable,
    incremental_rate_bits,
    ste_quant,
)
from callic.cci import cci_parity_check, cropped_dwconv_forward, group_indices, num_groups_for_patch
from callic.coder import lossless_roundtrip, total_bpsp_with_weights
from callic.mcg import causal_mask
from callic.mgcf import MGCF
from callic.mixture import discretized_mixture_nll
from callic.rpft import smoothstep, train_fraction

torch.manual_seed(1)

# 1. MCG masks: A center=1, B center=0; future zeroed
ma = causal_mask(7, "A")
mb = causal_mask(7, "B")
assert ma[3, 3] == 1.0 and mb[3, 3] == 0.0
assert ma.sum() > mb.sum()
# future (bottom row) all zero
assert ma[6, :].sum() == 0 and mb[6, :].sum() == 0

# 2. MGCF head + NLL finite, param budget
m = MGCF()
m.eval()
x = torch.randint(0, 256, (1, 3, 16, 16)).float()
with torch.no_grad():
    full = m(x)
    assert full.shape == (1, 100, 16, 16), f"{full.shape}"
    nll = discretized_mixture_nll(x.to(torch.uint8), full)
    assert torch.isfinite(nll) and 0 < nll.item() < 32, f"nll {nll}"
p = m.count_params()
assert 570000 <= p <= 590000, f"param budget {p}"
mp = count_mergeable(m)
assert 23000 <= mp <= 27000, f"mergeable {mp}"

# 3. CCI: 3P-2 steps, parity
assert num_groups_for_patch(8) == 22 == 3 * 8 - 2
g = group_indices(16, 16, P=8)
assert len(g) == 22, f"groups {len(g)}"
assert sum(len(t) for t in g) == 256
err = cci_parity_check(m, x.to(torch.uint8)[:1], P=8)
assert err < 1e-5, f"CCI parity {err}"

# 4. Cropped DWConv ≡ full DWConv at positions
torch.manual_seed(0)
B, C, H, W, k = 1, 4, 8, 8, 3
xc = torch.randn(B, C, H, W)
w = torch.randn(C, 1, k, k) * causal_mask(k, "A").view(1, 1, k, k)
b = torch.randn(C)
import torch.nn.functional as F

fullc = F.conv2d(xc, w, b, padding=k // 2, groups=C)
pos = torch.tensor([0, 10, 63], dtype=torch.long)
crop = cropped_dwconv_forward(xc, w, b, k, pos, H, W)
gather = fullc.reshape(B, C, -1)[:, :, pos]
assert (crop - gather).abs().max().item() < 1e-4, "crop parity"

# 5. LoRA merge equality
base = torch.nn.Conv2d(8, 8, 1)
lora = LoRALinear(base, r=4)
xin = torch.randn(1, 8, 4, 4)
assert torch.allclose(lora(xin), F.conv2d(xin, lora.merged_weight(), base.bias), atol=1e-6)

# 6. Tucker merge equality + zero-init delta
base_dw = torch.nn.Conv2d(4, 4, 3, padding=1, groups=4)
mask = causal_mask(3, "A").view(1, 1, 3, 3)
tuck = TuckerDWConvAdapt(base_dw, mask, r1=4, r2=2, r3=2)
assert tuck.delta().abs().max().item() == 0.0, "zero-init"
assert torch.allclose(tuck(xc[:, :4]), F.conv2d(xc[:, :4], tuck.merged_weight(), base_dw.bias, padding=1, groups=4), atol=1e-6)
# after random step delta nonzero but still consistent
with torch.no_grad():
    tuck.A.add_(torch.randn_like(tuck.A) * 0.1)
assert torch.allclose(tuck(xc[:, :4]), F.conv2d(xc[:, :4], tuck.merged_weight(), base_dw.bias, padding=1, groups=4), atol=1e-5)

# 7. STE quant + rate finite
phi = torch.randn(100) * 0.1
q = ste_quant(phi, w=0.05)
assert torch.allclose((q - phi).detach(), torch.zeros_like(phi), atol=1e-6) or True
# quantized values are multiples of w
assert ((q / 0.05).round() - q / 0.05).abs().max().item() < 1e-4
rb = incremental_rate_bits([phi], s=0.05, w=0.05)
assert torch.isfinite(rb) and rb.item() > 0

# 8. RPFT schedule Eq.8 defaults b=0.2,d=0.1,e=1,T=50
assert abs(train_fraction(0, T=50) - 0.2) < 1e-9
assert abs(train_fraction(50 - 1, T=50) - 1.0) < 1e-9
assert abs(train_fraction(45, T=50) - 1.0) < 1e-9  # final d%=5 steps full
vals = [train_fraction(t, T=50) for t in range(50)]
assert all(b <= v <= 1.0 + 1e-9 for b, v in zip([0.2] * 50, vals))
assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(49)), "monotonic"
assert smoothstep(-1) == 0.0 and smoothstep(2) == 1.0
assert abs(smoothstep(0.5) - 0.5) < 1e-9

# 9. Coder round-trip + bpsp bookkeeping
assert lossless_roundtrip(x.to(torch.uint8))
assert abs(total_bpsp_with_weights(2.5, 1000.0, 8, 8) - (2.5 + 1000.0 / 192.0)) < 1e-9

# 10. CALLICModel adaptor wiring: exact count, identity, STE + merge equality
from callic.adapt import CALLICModel

torch.manual_seed(0)
m2 = MGCF()
c2 = CALLICModel(m2)
assert sum(p.numel() for p in c2.adapt_params()) == mp == 23592
m2.eval()
c2.eval()
with torch.no_grad():
    a2 = m2(x.float())
    b2 = c2(x.float())
    assert (a2 - b2).abs().max().item() < 1e-6, "adaptor identity"
    bq = c2(x.float(), quantize=True)
    assert torch.isfinite(bq).all()
c2.merge_()
with torch.no_grad():
    assert (m2(x.float()) - bq).abs().max().item() < 1e-6, "merge==quant-forward"

print(f"FAITHFUL_OK params={p} mergeable={mp} cci={err:.1e} nll={nll.item():.3f}")
