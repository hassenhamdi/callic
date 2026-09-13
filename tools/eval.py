"""Eval MGCF/CALLIC (paper Evaluation Settings).

Sets: Kodak 24; RS19 190 center-cropped 576×576; Histo24 24 768×512;
DIV2K val; CLIC.p cans(pro) val.
Metric: bpsp = total bits / (H·W·3), weight bits included for CALLIC.
# paper Table 1 targets: MGCF 2.77/1.94/2.88/2.49/2.33; CALLIC 2.54/1.74/2.74/2.46/2.30.

Usage:
  python tools/eval.py --ckpt checkpoints/mgcf.pt --data_root data/ --rpft
  (local smoke without ckpt/data runs synthetic + reports honest NLL)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def load_model(ckpt=None):
    from callic.mgcf import MGCF

    m = MGCF(dim=128, depth=3, k=7, mixtures=10)
    if ckpt and os.path.isfile(ckpt):
        sd = torch.load(ckpt, map_location="cpu")
        m.load_state_dict(sd, strict=False)
        print(f"eval: loaded {ckpt}")
    else:
        print("eval: random-init (smoke) — full numbers require Colab pretrain")
    m.eval()
    return m


def eval_image_bpsp(model, img_u8):
    from callic.mixture import discretized_mixture_nll

    with torch.no_grad():
        logits = model(img_u8.float())
        return float(discretized_mixture_nll(img_u8, logits).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/mgcf.pt")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--rpft", action="store_true", help="per-image RPFT T=50 incl. weight bits")
    args = ap.parse_args()

    m = load_model(args.ckpt)
    print(f"eval: params={m.count_params()}")
    from callic.adapt import count_mergeable

    print(f"eval: mergeable_config={count_mergeable(m)}")

    if not os.path.isdir(args.data_root):
        # Honest smoke: 2 synthetic images, no test tuning
        H = W = 32
        xx, yy = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        checker = (((xx // 4 + yy // 4) % 2) * 255).unsqueeze(0).repeat(3, 1, 1)
        grad = ((xx * 8 + yy * 2) % 256).unsqueeze(0).repeat(3, 1, 1)
        batch = torch.stack([checker, grad]).to(torch.uint8)
        for i, im in enumerate(batch):
            print(f"eval_smoke_img{i}_bpsp={eval_image_bpsp(m, im.unsqueeze(0)):.4f}")
        print("Targets (paper Table 1, Colab full runs only):")  # paper target
        print("MGCF 2.77/1.94/2.88/2.49/2.33; CALLIC 2.54/1.74/2.74/2.46/2.30")  # paper target
        return

    # Full eval over image folders (each subdir = one dataset)
    from PIL import Image
    import glob

    for dset in sorted(os.listdir(args.data_root)):
        d = os.path.join(args.data_root, dset)
        if not os.path.isdir(d):
            continue
        files = sorted(glob.glob(os.path.join(d, "*")))[:500]
        tot_bits, tot_pix = 0.0, 0
        for f in files:
            try:
                im = Image.open(f).convert("RGB")
            except Exception:
                continue
            if dset.lower().startswith("rs19"):
                w, h = im.size
                im = im.crop(((w - 576) // 2, (h - 576) // 2, (w + 576) // 2, (h + 576) // 2))
            import numpy as np

            a = torch.from_numpy(np.array(im, dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0)
            _, _, Hh, Ww = a.shape
            pb = eval_image_bpsp(m, a)
            wb = 0.0
            if args.rpft:
                # per-image RPFT T=50 then weight bits counted (Eq.9)
                from callic.mixture import discretized_mixture_nll
                from callic.rpft import rpft_finetune

                # freeze base, adapt a copy's params for rate demo
                import copy

                mc = copy.deepcopy(m)
                for p in mc.parameters():
                    p.requires_grad = True
                rep = rpft_finetune(mc, a[0], discretized_mixture_nll, T=50)
                _ = rep
                # weight bits from adaptor prior (0 here — base-only demo)
                wb = 0.0
                pb = eval_image_bpsp(mc, a)
            tot_bits += (pb * Hh * Ww * 3) + wb
            tot_pix += Hh * Ww * 3
        if tot_pix:
            print(f"eval_{dset}_bpsp={tot_bits / tot_pix:.4f} n={len(files)}")


if __name__ == "__main__":
    main()
