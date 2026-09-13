"""Benchmark harness for CALLIC training speedups.

Covers every suggestion so far, each toggleable and timed the same way:
  system  : amp, channels-last, compile modes, cudnn.benchmark,
            matmul precision, fused Adam, fused (model+loss) graph, batch size
  schedule: warmup + peak LR, cosine vs OneCycle
  optimizer (OFFICIAL code only — no reimplemented math):
    adamw   : torch.optim.Adam (baseline)
    muon    : KellerJordan/Muon MuonWithAuxAdam (MIT)
              pip install git+https://github.com/KellerJordan/Muon
    normuon : zichongli5/NorMuon SingleDeviceNorMuonWithAuxAdam (MIT)
              pip install git+https://github.com/zichongli5/NorMuon.git
    aurora  : tilde-research/aurora-release aurora() (MIT, functional API;
              wrapped in a thin torch Optimizer that only manages momentum
              buffers and 2D reshaping — the update math stays official)
              git clone https://github.com/tilde-research/aurora-release.git
              then add its src/ to PYTHONPATH (no PyPI package exists)
  Turbo-Muon is intentionally absent: no official PyTorch implementation
  exists (AOL preconditioning lives in JAX optax only), so per repo policy
  it is not reimplemented here.

Usage:
  from tools.bench import BenchData, run_cfg, SYSTEM_PRESETS
  data = BenchData('data/DIV2K_valid_HR', n_patches=2048)
  print(run_cfg(data, steps=30, bs=32, compile_mode='default'))
  print(run_cfg(data, steps=100, bs=32, opt='muon'))
"""

import time

import torch


def split_params(model):
    """Routing (config, not math): Muon-eligible vs aux.

    Follows Keller Jordan's ConvNet guidance: Muon optimizes all
    convolutional filters except the first one on pixels (our 3×3 Type-B
    embedding); biases, LayerNorms, and the mixture head use AdamW.
    Same routing feeds Muon, NorMuon, and Aurora for a fair shootout.
    """
    muon, aux = [], []
    for mod in model.modules():
        if isinstance(mod, torch.nn.LayerNorm):
            aux += list(mod.parameters())
    seen = {id(p) for p in aux}
    for n, p in model.named_parameters():
        if id(p) in seen:
            continue
        if p.ndim >= 2 and "embed" not in n and "head" not in n:
            muon.append(p)
        else:
            aux.append(p)
    return muon, aux


class _AuroraWrapper(torch.optim.Optimizer):
    """Thin torch-Optimizer shell around official tilde-research aurora().

    Only plumbing is local: caller-managed momentum buffers and 2D reshaping
    of conv filters ([out, rest], same reshape NorMuon uses). The update math
    (damped alternating row-norm + polar) is 100% official aurora().
    Non-2D-routable params (biases, norms, head) go to an internal AdamW.
    """

    def __init__(self, muon_params, aux_params, lr=0.05, aux_lr=3e-4,
                 weight_decay=0.025, mu=0.95, pp_iterations=2, pp_beta=0.5):
        import os as _os
        import sys as _sys
        try:
            from aurora import aurora as _aurora
        except ImportError:
            # No PyPI package exists; accept an official git clone on disk.
            cands = [_os.path.join(_os.getcwd(), "thirdparty", "aurora-release", "src"),
                     _os.environ.get("AURORA_SRC", "")]
            for c in cands:
                if c and _os.path.isdir(c) and c not in _sys.path:
                    _sys.path.insert(0, c)
            try:
                from aurora import aurora as _aurora
            except ImportError as e:
                raise ImportError(
                    "aurora package not found: git clone "
                    "https://github.com/tilde-research/aurora-release.git "
                    "thirdparty/aurora-release (or set AURORA_SRC=.../src)") from e
        self._aurora = _aurora
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu,
                        pp_iterations=pp_iterations, pp_beta=pp_beta)
        super().__init__(muon_params, defaults)
        self.aux = torch.optim.AdamW(aux_params, lr=aux_lr) if aux_params else None

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                o = p.shape[0]
                if "mom" not in st:
                    st["mom"] = torch.zeros(o, p.numel() // o,
                                            device=p.device, dtype=torch.float32)
                # Clone: official aurora() mutates W (and G via lerp_) in place,
                # so it must not alias p.data / p.grad (copy_ self-assign errors).
                W2 = p.detach().reshape(o, -1).clone()
                G2 = p.grad.detach().reshape(o, -1).float().clone()
                self._aurora(W2, G2, st["mom"], eta=group["lr"],
                             weight_decay=group["weight_decay"], mu=group["mu"],
                             pp_iterations=group["pp_iterations"],
                             pp_beta=group["pp_beta"])
                p.data.copy_(W2.reshape(p.shape).to(p.dtype))
        if self.aux:
            self.aux.step()

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        if self.aux:
            self.aux.zero_grad(set_to_none=set_to_none)


def _xla_device():
    """TPU device via torch_xla, or None. Import is lazy (package rarely installed)."""
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    except ImportError as e:
        raise ImportError(
            "TPU requested but torch_xla not installed: pip install torch_xla "
            "(see https://docs.pytorch.org/xla/ for the TPU wheel)") from e


def build_optimizer(name, model, lr=5e-4, muon_lr=0.02, weight_decay=0.0):
    """Official optimizers only. Raises with install instructions if missing."""
    mp, ap = split_params(model)
    if name == "adamw":
        try:
            return torch.optim.Adam(model.parameters(), lr=lr, fused=True)
        except Exception:
            return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "muon":
        try:
            from muon import SingleDeviceMuonWithAuxAdam
        except ImportError as e:
            raise ImportError(
                "KellerJordan Muon not found: pip install "
                "git+https://github.com/KellerJordan/Muon") from e
        return SingleDeviceMuonWithAuxAdam([
            dict(params=mp, use_muon=True, lr=muon_lr, weight_decay=weight_decay),
            dict(params=ap, use_muon=False, lr=lr, betas=(0.9, 0.95),
                 weight_decay=weight_decay),
        ])
    if name == "normuon":
        try:
            from normuon import SingleDeviceNorMuonWithAuxAdam
        except ImportError as e:
            raise ImportError(
                "NorMuon not found: pip install "
                "git+https://github.com/zichongli5/NorMuon.git") from e
        return SingleDeviceNorMuonWithAuxAdam([
            dict(params=mp, use_muon=True, lr=muon_lr, weight_decay=weight_decay),
            dict(params=ap, use_muon=False, lr=lr, betas=(0.9, 0.95),
                 weight_decay=weight_decay),
        ])
    if name == "aurora":
        # eta default 0.05 is the paper value; pass muon_lr to sweep it
        # (0.02 matches the Muon shootout setting).
        return _AuroraWrapper(mp, ap, lr=muon_lr, aux_lr=lr)
    raise ValueError(f"unknown optimizer {name!r} (adamw|muon|normuon|aurora)")


class BenchData:
    """Pinned RAM patch pools with a HELD-OUT eval split by file.

    Train and eval patches come from disjoint images, so eval loss measures
    generalization — not memorization of the training pool. (Comparing
    in-loop training losses once overfitting starts is meaningless: a big
    step can collapse onto the pool and report near-zero training NLL.)
    """

    def __init__(self, path="data/DIV2K_valid_HR", n_patches=2048,
                 n_eval=256, P=64):
        import glob as _glob

        import numpy as np
        from PIL import Image

        files = sorted(_glob.glob(f"{path}/**/*.png", recursive=True))
        if not files:
            files = sorted(_glob.glob("data/eval/kodak/*.png"))
        self.patches, self.eval_patches = [], []
        self.source = files[0] if files else "synthetic"
        if files:
            cut = max(1, int(len(files) * 0.8))
            for fi, f in enumerate(files):
                a = np.array(Image.open(f).convert("RGB"), dtype=np.uint8)
                h, w, _ = a.shape
                pool = self.patches if fi < cut else self.eval_patches
                cap = n_patches if fi < cut else n_eval
                for y in range(0, h - P + 1, P):
                    for x in range(0, w - P + 1, P):
                        pool.append(
                            torch.from_numpy(a[y : y + P, x : x + P].transpose(2, 0, 1)))
                        if len(pool) >= cap:
                            break
                    if len(pool) >= cap:
                        break
                if len(self.patches) >= n_patches and len(self.eval_patches) >= n_eval:
                    break
        if not self.patches:  # synthetic fallback (timing only, not loss curves)
            xx, yy = torch.meshgrid(torch.arange(P), torch.arange(P), indexing="ij")
            chk = (((xx // 4 + yy // 4) % 2) * 255).unsqueeze(0).repeat(3, 1, 1)
            self.patches = [chk.to(torch.uint8)] * 64
            self.eval_patches = [chk.to(torch.uint8)] * 8
        self.data = torch.stack(self.patches)
        self.eval_data = torch.stack(self.eval_patches)

    def pin(self):
        try:
            self.data = self.data.pin_memory()
        except Exception:
            pass
        return self


def run_cfg(data, steps=30, bs=32, lr=5e-4, opt="adamw", warmup=0,
            amp=True, cl=True, compile_mode=None, cudnn_bench=False,
            matmul_high=False, fused_adam=True, fused_loss=False,
            muon_lr=0.02, xla_bf16=True, seed=0, device=None):
    """One timed config. Returns dict with s/step, patches/s, loss0→loss1.

    device: 'cuda' | 'cpu' | 'xla' (TPU via torch_xla). On XLA, CUDA-only
    flags (amp-fp16, channels-last, torch.compile, cudnn) are ignored by
    design: XLA compiles the graph itself, layout is handled by the
    compiler, and precision is bf16 (xla_bf16) with the NLL kept in fp32.
    Call xm.mark_step() semantics: loss.backward() + optimizer step are
    followed by torch_xla sync each iteration.
    """
    from callic.mgcf import MGCF
    from callic.mixture import discretized_mixture_nll

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device == "cuda"
    use_xla = device == "xla"
    xm = None
    if use_xla:
        import torch_xla.core.xla_model as xm
        device = _xla_device()
    if use_cuda and cudnn_bench:
        torch.backends.cudnn.benchmark = True
    if matmul_high and not use_xla:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    m = MGCF()
    if use_cuda and cl:
        m = m.to(memory_format=torch.channels_last)
    m = m.to(device)
    compiled = False
    if fused_loss:
        class _MLL(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, b):
                return discretized_mixture_nll(b, self.net(b.float()).float())

        target = _MLL(m)
    else:
        target = m
    fn = target
    if use_cuda and compile_mode:
        try:
            fn = torch.compile(target, mode=compile_mode)
            compiled = True
        except Exception:
            fn = target
    if opt == "adamw":
        _lr = lr
        try:
            optim = torch.optim.Adam(m.parameters(), lr=_lr, fused=use_cuda and fused_adam)
        except Exception:
            optim = torch.optim.Adam(m.parameters(), lr=_lr)
    elif opt in ("muon", "normuon", "aurora"):
        optim = build_optimizer(opt, m, lr=lr, muon_lr=muon_lr)
    else:
        raise ValueError(f"unknown optimizer {opt!r} (adamw|muon|normuon|aurora)")
    sched = None
    if warmup:
        sched = torch.optim.lr_scheduler.SequentialLR(
            optim,
            [torch.optim.lr_scheduler.LinearLR(optim, 1e-6, total_iters=warmup),
             torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, steps - warmup))],
            milestones=[warmup])
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and use_cuda))
    m.train()
    if use_cuda:
        data.pin()
    D, losses = data.data, []
    t0 = time.time()
    first = None
    for s in range(steps):
        if use_xla:
            b = D[torch.randint(0, len(D), (bs,))].to(device)
        else:
            b = D[torch.randint(0, len(D), (bs,))].to(device, non_blocking=True)
        if use_cuda and cl:
            b = b.to(memory_format=torch.channels_last)
        optim.zero_grad(set_to_none=True)
        if use_xla and xla_bf16:
            with torch.autocast("xla", dtype=torch.bfloat16):
                logits = fn(b.float())
            loss = discretized_mixture_nll(b, logits.float())
        elif fused_loss:
            with torch.amp.autocast("cuda", enabled=(amp and use_cuda)):
                fwd = fn(b)
            loss = fwd if torch.is_tensor(fwd) else fwd
        else:
            with torch.amp.autocast("cuda", enabled=(amp and use_cuda)):
                logits = fn(b.float())
            loss = discretized_mixture_nll(b, logits.float())
        if use_xla:
            import torch_xla.core.xla_model as _xm

            loss.backward()
            _xm.optimizer_step(optim)  # all-reduce + step + sync
        else:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        if sched is not None:
            sched.step()
        if s == 0:
            first = time.time() - t0
        if use_xla:
            import torch_xla.core.xla_model as _xm2

            _xm2.mark_step()
            losses.append(float(loss.detach().cpu().item()))
        else:
            losses.append(float(loss.item()))
    dt = time.time() - t0
    steady = (dt - (first or 0)) / max(1, steps - 1)
    # Held-out eval (disjoint images): the decision metric. No grad, fp32,
    # chunked so the eval forward never OOMs.
    m.eval()
    with torch.no_grad():
        from callic.mixture import discretized_mixture_nll as _nll

        tot, n = 0.0, 0
        for i in range(0, len(data.eval_data), 32):
            eb = data.eval_data[i : i + 32].to(device)
            tot += float(_nll(eb, m(eb.float()).float()).item()) * len(eb)
            n += len(eb)
        eval_loss = tot / max(1, n)
    return {"opt": opt, "bs": bs, "device": str(device),
            "amp": amp and use_cuda, "cl": cl and use_cuda,
            "compile": compile_mode if compiled else None, "cudnn_bench": cudnn_bench,
            "fused_loss": fused_loss, "warmup": warmup,
            "s_per_step": round(steady, 4), "patches_per_s": round(bs / steady, 1),
            "loss0": round(losses[0], 3), "lossN": round(losses[-1], 3),
            "evalN": round(eval_loss, 4)}


SYSTEM_PRESETS = [
    ("baseline", {}),
    ("+cudnn.benchmark", {"cudnn_bench": True}),
    ("+matmul-high", {"matmul_high": True}),
    ("+compile", {"compile_mode": "default"}),
    ("+compile-max", {"compile_mode": "max-autotune"}),
    ("+compile-cg", {"compile_mode": "reduce-overhead"}),
    ("+fused-loss", {"compile_mode": "default", "fused_loss": True}),
    ("all-system", {"cudnn_bench": True, "matmul_high": True,
                    "compile_mode": "default", "fused_loss": True}),
]
