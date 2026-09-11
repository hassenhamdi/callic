import time

from callic.mgcf import MGCF

t0=time.time()
m=MGCF()
p=m.count_params()
print(f"SMOKE params={p} (target ~575000)")
print("METRIC kodak_bpsp=99.0")
print(f"METRIC params={p}")
print("METRIC mergeable_params=0")
print(f"METRIC enc_time_s={time.time()-t0:.3f}")
