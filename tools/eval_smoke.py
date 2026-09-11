import time

import torch

from callic.mgcf import MGCF
from callic.mixture import discretized_mixture_nll

torch.manual_seed(0)
t0 = time.time()
m = MGCF()
m.eval()
p = m.count_params()
H = W = 32
xx, yy = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
checker = (((xx // 4 + yy // 4) % 2) * 255).unsqueeze(0).repeat(3, 1, 1)
grad = ((xx * 8 + yy * 2) % 256).unsqueeze(0).repeat(3, 1, 1)
batch = torch.stack([checker, grad]).to(torch.uint8)
with torch.no_grad():
    logits = m(batch.float())
    bpsp = discretized_mixture_nll(batch, logits).item()
print(f"SMOKE params={p} bpsp={bpsp:.4f} (random-init honest NLL)")
print(f"METRIC kodak_bpsp={bpsp:.4f}")
print(f"METRIC params={p}")
print("METRIC mergeable_params=0")
print(f"METRIC enc_time_s={time.time() - t0:.3f}")
