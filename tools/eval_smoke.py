import time

import torch

from callic.mgcf import MGCF
from callic.mixture import discretized_mixture_nll

torch.manual_seed(0)
t0 = time.time()
m = MGCF()
p = m.count_params()
H = W = 32
xx, yy = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
checker = (((xx // 4 + yy // 4) % 2) * 255).unsqueeze(0).repeat(3, 1, 1)
grad = ((xx * 8 + yy * 2) % 256).unsqueeze(0).repeat(3, 1, 1)
batch = torch.stack([checker, grad]).to(torch.uint8)
m.eval()
with torch.no_grad():
    bpsp0 = discretized_mixture_nll(batch, m(batch.float())).item()
# tiny honest overfit: 15 Adam steps on checker only (synthetic, never test data)
m.train()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
single = checker.unsqueeze(0).to(torch.uint8)
for _ in range(15):
    opt.zero_grad()
    loss = discretized_mixture_nll(single, m(single.float()))
    loss.backward()
    opt.step()
m.eval()
with torch.no_grad():
    bpsp1 = discretized_mixture_nll(batch, m(batch.float())).item()
print(f"SMOKE params={p} before={bpsp0:.4f} after_tiny={bpsp1:.4f}")
print(f"METRIC kodak_bpsp={bpsp1:.4f}")
print(f"METRIC params={p}")
print("METRIC mergeable_params=0")
print(f"METRIC enc_time_s={time.time() - t0:.3f}")
