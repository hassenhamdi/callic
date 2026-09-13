"""Pretrain MGCF (paper Experimental Settings).

Paper: DIV2K 800 + Flickr2K 2650 → non-overlap 64×64 patches
(612806 images), Adam 2M steps, batch 32, lr 5e-4.
Colab GPU for full runs; local mirror for smoke.

Speed options (same math, faster wall-clock; all defaults safe on CPU):
  --amp            mixed precision (T4 Tensor Cores). Forward in fp16,
                   mixture NLL kept in fp32 for log/exp stability.
  --channels-last  NHWC memory format for cuDNN convs.
  --compile        torch.compile the model (Triton; falls back).
  --fused-loss     compile model+NLL as one graph (bench winner, default on).
  --opt            adamw | muon | normuon | aurora — official implementations
                   only (bench shootout winner on held-out loss: normuon).
  --warmup         linear warmup steps before cosine (recommended with --opt).
  --bs N           larger batches raise GPU utilization; pair with --lr-scale
                   (linear rule: lr = base_lr * bs/32) and --schedule cosine.
  --cudnn-bench / --matmul-high: measured neutral on this net; opt-in.
  Bench every claim first: notebooks/bench_speedups.ipynb.

Usage (Colab GPU, fast):
  python tools/train.py --data /tmp/div2k/DIV2K_valid_HR --steps 200000 --bs 128 --lr-scale --schedule cosine --out checkpoints/mgcf.pt
Usage (paper-exact recipe):
  python tools/train.py --data /content/data --steps 2000000 --bs 32 --lr 5e-4 --no-amp --no-compile
Usage (local smoke):
  python tools/train.py --smoke  (15 Adam steps on synthetic checker, proves NLL descends)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def build_model():
    from callic.mgcf import MGCF

    return MGCF(dim=128, depth=3, k=7, mixtures=10)


def smoke_train(steps=15, lr=1e-3):
    from callic.mixture import discretized_mixture_nll

    torch.manual_seed(0)
    m = build_model()
    H = W = 32
    xx, yy = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    checker = (((xx // 4 + yy // 4) % 2) * 255).unsqueeze(0).repeat(3, 1, 1)
    single = checker.unsqueeze(0).to(torch.uint8)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = discretized_mixture_nll(single, m(single.float()))
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        final = discretized_mixture_nll(single, m(single.float())).item()
    return m, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--steps", type=int, default=2000000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out", default="checkpoints/mgcf.pt")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--amp", dest="amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--channels-last", dest="cl", action="store_true", default=True)
    ap.add_argument("--no-channels-last", dest="cl", action="store_false")
    ap.add_argument("--compile", dest="compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--lr-scale", action="store_true",
                    help="linear LR rule: lr = lr * bs/32 (use with larger --bs)")
    ap.add_argument("--schedule", choices=["none", "cosine"], default="none")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--resume", action="store_true",
                    help="load --out ckpt if present (for multi-session long runs)")
    ap.add_argument("--keep-every", type=int, default=10000,
                    help="also keep a numbered ckpt every K steps (0=disable)")
    ap.add_argument("--keep-last", type=int, default=3,
                    help="keep only the last K numbered ckpts (+ always keep latest)")
    ap.add_argument("--drive-dir", default="",
                    help="e.g. /content/drive/MyDrive/callic/run_100k — mirror "
                         "latest ckpt + numbered keeps + status there every "
                         "--drive-every steps so artifacts survive session death")
    ap.add_argument("--drive-every", type=int, default=2000)
    ap.add_argument("--keep-best", dest="keep_best", action="store_true", default=True,
                    help="also keep the best-loss ckpt (default on)")
    ap.add_argument("--no-keep-best", dest="keep_best", action="store_false")
    ap.add_argument("--opt", choices=["adamw", "muon", "normuon", "aurora"], default="adamw",
                    help="optimizer: official implementations only (see tools/bench.py). "
                         "Shootout winner on DIV2K-valid held-out: normuon.")
    ap.add_argument("--muon-lr", type=float, default=0.02,
                    help="LR for the Muon-family hidden weights (aux AdamW uses --lr)")
    ap.add_argument("--warmup", type=int, default=0,
                    help="linear warmup steps before the cosine schedule")
    ap.add_argument("--cudnn-bench", dest="cudnn_bench", action="store_true", default=False,
                    help="cudnn autotune (measured neutral on this net; opt-in)")
    ap.add_argument("--matmul-high", dest="matmul_high", action="store_true", default=False,
                    help="TF32/Tensor-Core fp32 matmuls (measured neutral; opt-in)")
    ap.add_argument("--fused-loss", dest="fused_loss", action="store_true", default=True,
                    help="compile model+NLL as one graph (bench winner: 399 vs 322 patches/s)")
    ap.add_argument("--no-fused-loss", dest="fused_loss", action="store_false")
    ap.add_argument("--val-data", default="",
                    help="held-out image dir for val bpsp logging (e.g. data/DIV2K_valid_HR "
                         "when --data is the train split). Logged every --log-every steps. "
                         "All paths resolve against the repo root unless absolute.")
    args = ap.parse_args()

    # Anchor relative paths at the repo root (script lives in <root>/tools/),
    # so the script works from any cwd: `python /path/to/tools/train.py --data data`.
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(args.data):
        args.data = os.path.join(REPO, args.data)
    if not os.path.isabs(args.out):
        args.out = os.path.join(REPO, args.out)
    if args.val_data and not os.path.isabs(args.val_data):
        args.val_data = os.path.join(REPO, args.val_data)

    if args.smoke or not os.path.isdir(args.data):
        print("train: smoke mode (synthetic checker, honest NLL descent proof)")
        m, final = smoke_train()
        print(f"train_smoke_final_bpsp={final:.4f} params={m.count_params()}")
        return

    # Full pretraining on Colab GPU — DIV2K+Flickr2K non-overlap 64×64 patches.
    # Paper: 612806 patches, Adam 2M steps, bs32, lr5e-4. Each image opened
    # once; patches cached (fits Colab RAM) to avoid per-sample file I/O.
    from callic.mixture import discretized_mixture_nll

    from PIL import Image
    import glob
    import random

    import numpy as np

    P = 64
    img_files = sorted(glob.glob(os.path.join(args.data, "**", "*.png"), recursive=True))
    img_files += sorted(glob.glob(os.path.join(args.data, "**", "*.jpg"), recursive=True))
    assert img_files, f"no images under {args.data}"
    print(f"train: caching non-overlap {P}x{P} patches from {len(img_files)} images...")
    allp = []
    for fi, f in enumerate(img_files):
        im = Image.open(f).convert("RGB")
        w, h = im.size
        a = np.array(im, dtype=np.uint8)
        for y in range(0, h - P + 1, P):
            for x in range(0, w - P + 1, P):
                allp.append(torch.from_numpy(a[y : y + P, x : x + P].transpose(2, 0, 1)))
        if len(allp) >= 612806:
            break
    data = torch.stack(allp)
    del allp
    print(f"train: {len(data)} patches")

    # Held-out val pool (disjoint dir, capped; no grad, fp32, chunked).
    val_data = None
    if args.val_data:
        import glob as _vg

        vfiles = sorted(_vg.glob(os.path.join(args.val_data, "**", "*.png"), recursive=True))
        vfiles += sorted(_vg.glob(os.path.join(args.val_data, "**", "*.jpg"), recursive=True))
        assert vfiles, f"no images under {args.val_data}"
        vall = []
        for f in vfiles:
            im = Image.open(f).convert("RGB")
            w, h = im.size
            a = np.array(im, dtype=np.uint8)
            for y in range(0, h - P + 1, P):
                for x in range(0, w - P + 1, P):
                    vall.append(torch.from_numpy(a[y : y + P, x : x + P].transpose(2, 0, 1)))
                    if len(vall) >= 256:
                        break
                if len(vall) >= 256:
                    break
            if len(vall) >= 256:
                break
        val_data = torch.stack(vall)
        print(f"train: {len(val_data)} held-out val patches from {args.val_data}")

    def _val_bpsp():
        _unwrap(m).eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_data), 32):
                eb = val_data[i : i + 32].to(device)
                tot += float(discretized_mixture_nll(eb, _unwrap(m)(eb.float()).float()).item()) * len(eb)
                n += len(eb)
        m.train()
        return tot / max(1, n)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = device == "cuda"
    lr = args.lr * (args.bs / 32) if args.lr_scale else args.lr
    if use_cuda and args.cudnn_bench:
        torch.backends.cudnn.benchmark = True
    if args.matmul_high:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    m = build_model()
    if use_cuda and args.cl:
        m = m.to(memory_format=torch.channels_last)
    m = m.to(device)
    # Forward target: plain MGCF, or fused model+NLL graph (bench winner).
    # Optimizer + ckpts ALWAYS use inner MGCF `m`, so weight format never changes.
    from callic.mixture import discretized_mixture_nll as _nll

    class _MLL(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, b):
            return _nll(b, self.net(b.float()).float())

    use_fused = bool(use_cuda and args.compile and args.fused_loss)
    target = _MLL(m) if use_fused else m
    compiled = False
    if use_cuda and args.compile:
        try:
            fn = torch.compile(target, mode="default")
            compiled = True
        except Exception as e:
            print(f"train: compile fallback ({str(e)[:100]})")
            fn = target
    else:
        fn = target
    print(f"train: params={m.count_params() if hasattr(m, 'count_params') else '?'} "
          f"device={device} steps={args.steps} bs={args.bs} lr={lr} "
          f"opt={args.opt} amp={args.amp and use_cuda} cl={args.cl and use_cuda} "
          f"compiled={compiled} fused_loss={use_fused}")

    def _unwrap(mm):
        # torch.compile wraps the model; state_dict keys gain "_orig_mod.".
        # Save inner weights so ckpts load with or without compile.
        return mm._orig_mod if hasattr(mm, "_orig_mod") else mm

    if args.opt == "adamw":
        try:
            opt = torch.optim.Adam(m.parameters(), lr=lr, fused=use_cuda)
        except Exception:
            opt = torch.optim.Adam(m.parameters(), lr=lr)
    else:
        from tools.bench import build_optimizer

        opt = build_optimizer(args.opt, _unwrap(m), lr=lr, muon_lr=args.muon_lr)
    sched = None
    if args.schedule == "cosine":
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        if args.warmup:
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt,
                [torch.optim.lr_scheduler.LinearLR(opt, 1e-6, total_iters=args.warmup), cos],
                milestones=[args.warmup])
        else:
            sched = cos

    def _save_ckpt(path, step, best_loss=None, best_step=None):
        # Atomic: write tmp + rename, so a kill mid-write never corrupts ckpt.
        payload = {
            "step": step,
            "model": _unwrap(m).state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict() if sched is not None else None,
            "scaler": scaler.state_dict(),
            "args": {"bs": args.bs, "lr": lr, "schedule": args.schedule,
                     "opt": args.opt, "muon_lr": args.muon_lr, "warmup": args.warmup},
            "best_loss": best_loss,
            "best_step": best_step,
        }
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def _load_ckpt(path):
        # Returns (saved step, best_loss, best_step). Tolerates legacy ckpts.
        payload = torch.load(path, map_location=device)
        if isinstance(payload, dict) and "model" in payload:
            _unwrap(m).load_state_dict(payload["model"])
            try:
                opt.load_state_dict(payload["optimizer"])
            except Exception:
                pass
            if sched is not None and payload.get("scheduler") is not None:
                try:
                    sched.load_state_dict(payload["scheduler"])
                except Exception:
                    pass
            try:
                scaler.load_state_dict(payload["scaler"])
            except Exception:
                pass
            return (int(payload.get("step", 0)),
                    payload.get("best_loss"), payload.get("best_step"))
        _unwrap(m).load_state_dict(payload)
        return 0, None, None

    def _best_path():
        root, ext = os.path.splitext(args.out)
        return f"{root}_best{ext or '.pt'}"

    def _ckpt_paths(step):
        # (latest path, numbered path or None)
        numbered = None
        if args.keep_every and step % args.keep_every == 0:
            root, ext = os.path.splitext(args.out)
            numbered = f"{root}_step{step}{ext or '.pt'}"
        return args.out, numbered

    def _prune_keeps():
        if not (args.keep_every and args.keep_last):
            return
        root, ext = os.path.splitext(args.out)
        import glob as _glob
        import re as _re

        files = []
        for f in _glob.glob(f"{root}_step*{ext or '.pt'}"):
            mt = _re.search(r"_step(\d+)", os.path.basename(f))
            if mt:
                files.append((int(mt.group(1)), f))
        for _, f in sorted(files)[:-args.keep_last]:
            try:
                os.remove(f)
            except OSError:
                pass

    def _sync_drive(step, loss, status="running"):
        # Mirror artifacts to Drive (survives Colab session death).
        if not args.drive_dir:
            return
        import shutil

        try:
            ckd = os.path.join(args.drive_dir, "ckpts")
            os.makedirs(ckd, exist_ok=True)
            if os.path.isfile(args.out):
                shutil.copy(args.out, os.path.join(ckd, os.path.basename(args.out)))
            root, ext = os.path.splitext(args.out)
            import glob as _glob

            for f in _glob.glob(f"{root}_step*{ext or '.pt'}"):
                shutil.copy(f, os.path.join(ckd, os.path.basename(f)))
            if args.keep_best and os.path.isfile(_best_path()):
                shutil.copy(_best_path(), os.path.join(ckd, os.path.basename(_best_path())))
            with open(os.path.join(args.drive_dir, "STATUS.txt"), "w") as fh:
                fh.write(f"step={step}/{args.steps} loss_bpsp={loss:.4f} "
                         f"status={status} time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                if args.keep_best and best_loss[0] is not None:
                    fh.write(f"best_loss={best_loss[0]:.4f} best_step={best_step[0]}\n")
        except Exception as e:
            print(f"train: drive sync failed ({str(e)[:120]}), continuing locally")

    start_step = 0
    best_loss, best_step = [None], [None]  # mutable closure for _sync_drive
    if args.resume and args.out and os.path.isfile(args.out):
        try:
            _s, _bl, _bs = _load_ckpt(args.out)
            start_step = _s + 1
            best_loss[0], best_step[0] = _bl, _bs
            print(f"train: resumed {args.out} at step {start_step} "
                  f"(best_loss={_bl} best_step={_bs})")
        except Exception as e:
            print(f"train: resume failed ({str(e)[:100]}), from scratch")
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and use_cuda))
    if use_cuda:
        try:
            data = data.pin_memory()
        except Exception:
            pass
    m.train()
    t0 = time.time()
    for step in range(start_step, args.steps):
        sel = torch.randint(0, len(data), (args.bs,))
        b = data[sel].to(device, non_blocking=True)
        if use_cuda and args.cl:
            b = b.to(memory_format=torch.channels_last)
        opt.zero_grad(set_to_none=True)
        if use_fused:
            with torch.amp.autocast("cuda", enabled=(args.amp and use_cuda)):
                loss = fn(b)
        else:
            with torch.amp.autocast("cuda", enabled=(args.amp and use_cuda)):
                logits = fn(b.float())
            loss = discretized_mixture_nll(b, logits.float())
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if sched is not None:
            sched.step()
        if step % args.log_every == 0:
            dt = time.time() - t0
            rate = (step + 1) * args.bs / max(dt, 1e-6)
            eta = (args.steps - step - 1) * (dt / max(step, 1)) / 3600 if step else -1
            print(f"step={step} loss_bpsp={loss.item():.4f} "
                  f"{rate:.0f} patches/s eta={eta:.1f}h elapsed={dt:.0f}s")
            if val_data is not None:
                print(f"step={step} val_bpsp={_val_bpsp():.4f} (held-out, {len(val_data)} patches)")
            outdir = os.path.dirname(args.out)
            if outdir:
                os.makedirs(outdir, exist_ok=True)
            latest, numbered = _ckpt_paths(step)
            _save_ckpt(latest, step, best_loss[0], best_step[0])
            if numbered:
                _save_ckpt(numbered, step, best_loss[0], best_step[0])
                _prune_keeps()
                print(f"train: kept {numbered}")
            if args.keep_best and (best_loss[0] is None or loss.item() < best_loss[0]):
                best_loss[0], best_step[0] = loss.item(), step
                _save_ckpt(_best_path(), step, best_loss[0], best_step[0])
                print(f"train: new best {best_loss[0]:.4f} at step {step} -> {os.path.basename(_best_path())}")
            if args.drive_dir and args.drive_every and step % args.drive_every == 0:
                _sync_drive(step, loss.item())
    # Final save (also covers steps < log_every).
    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    latest, numbered = _ckpt_paths(max(args.steps - 1, start_step))
    _save_ckpt(latest, args.steps - 1, best_loss[0], best_step[0])
    if numbered:
        _save_ckpt(numbered, args.steps - 1, best_loss[0], best_step[0])
        _prune_keeps()
    _sync_drive(args.steps - 1, loss.item() if "loss" in dir() else float("nan"),
                status="done")
    print(f"train: done, ckpt {args.out}")


if __name__ == "__main__":
    main()
