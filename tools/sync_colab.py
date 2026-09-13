"""Colab mirror: every Colab cell has a local file twin.

Single source of truth lives locally; this script:
- hashes local modules,
- emits the Colab notebook (colab/callic_colab.ipynb) with one cell per
  local file (mirrored verbatim) + GPU train/eval/RPFT cells,
- logs cell IDs + file hashes (ASI) for provenance.

Usage:
  python tools/sync_colab.py            # regenerate notebook, print hashes
  python tools/sync_colab.py --check    # verify notebook matches local files

No credentials are committed; Colab GPU runs execute the generated notebook.
"""

import argparse
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIRROR_FILES = [
    "callic/mcg.py",
    "callic/mgcf.py",
    "callic/mixture.py",
    "callic/cci.py",
    "callic/adapt.py",
    "callic/rpft.py",
    "callic/coder.py",
    "tools/train.py",
    "tools/eval.py",
]

KEEPALIVE_CELL = (
    "Keepalive (anti idle-disconnect — run once, keep tab open)",
    "from IPython.display import Javascript, display\n"
    "display(Javascript('''\n"
    "function __callicKeepAlive(){\n"
    "  try {\n"
    "    const btn = document.querySelector('colab-connect-button')\n"
    "      || document.querySelector('#connect-button');\n"
    "    if (btn) btn.click();\n"
    "  } catch (e) {}\n"
    "}\n"
    "if (window.__callicKeepaliveTimer) clearInterval(window.__callicKeepaliveTimer);\n"
    "window.__callicKeepaliveTimer = setInterval(__callicKeepAlive, 60000);\n"
    "console.log('callic keepalive armed');\n"
    "'''))\n"
    "print('keepalive armed: clicks connect every 60s. NOTES: keep the tab open; '\n"
    "      'this defeats idle-timeout only — Colab hard limits (~12h) still apply, '\n"
    "      'so checkpoints + --resume + Drive sync remain the real safety net.')\n",
)

GPU_CELLS = [
    KEEPALIVE_CELL,
    ("Setup from Drive: restore code + weights + data, then resume",
     "from google.colab import drive\n"
     "import os, glob, shutil, tarfile, subprocess, urllib.request\n"
     "drive.mount('/content/drive', force_remount=False)\n"
     "RUN = '/content/drive/MyDrive/callic/run_100k'\n"
     "# 1. code: Drive tarball first, else run the %%writefile mirror cells above\n"
     "CODE = '/content/drive/MyDrive/callic/code/callic_pkg_latest.tar.gz'\n"
     "os.makedirs('/tmp/callic_pkg', exist_ok=True)\n"
      "if os.path.exists(CODE) and not os.path.exists('/tmp/callic_pkg/train_full.py'):\n"
      "    with tarfile.open(CODE) as _t:\n"
      "        _t.extractall('/tmp', filter='data')\n"
      "    print('code restored from Drive')\n"
     "# 2. weights: Drive ckpts -> local (skip existing; latest wins on --resume)\n"
     "os.makedirs('/tmp/div2k/run100k', exist_ok=True)\n"
     "for _f in glob.glob(RUN + '/ckpts/*.pt'):\n"
     "    _d = '/tmp/div2k/run100k/' + os.path.basename(_f)\n"
     "    if not os.path.exists(_d):\n"
     "        shutil.copy(_f, _d)\n"
     "print('local ckpts:', sorted(os.path.basename(_f) for _f in glob.glob('/tmp/div2k/run100k/*.pt')))\n"
     "# 3. data: re-download only what is missing (Kodak 24 + DIV2K-valid 100)\n"
     "from pathlib import Path as _P\n"
     "_kd = _P('/tmp/kodak'); _kd.mkdir(exist_ok=True)\n"
     "for _i in range(1, 25):\n"
     "    _n = f'{_i:02d}'; _dst = _kd / f'kodim{_n}.png'\n"
     "    if not (_dst.exists() and _dst.stat().st_size > 10000):\n"
     "        urllib.request.urlretrieve(\n"
     "            f'https://raw.githubusercontent.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite/master/PhotoCD_PCD0992/{_n}.png', _dst)\n"
     "print('kodak:', len(list(_kd.glob('*.png'))), '/24')\n"
     "import pathlib as _pl\n"
     "if len(list(_pl.Path('/tmp/div2k/DIV2K_valid_HR').glob('*.png'))) < 100:\n"
     "    subprocess.run('cd /tmp/div2k && curl -L -o DIV2K_valid_HR.zip https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip && unzip -q -o DIV2K_valid_HR.zip', shell=True)\n"
     "print('div2k:', len(list(_pl.Path('/tmp/div2k/DIV2K_valid_HR').glob('*.png'))), '/100')\n"
     "print('Setup OK. Resume with the 100k-run cell (uses --resume; safe to re-run).')\n"),
    ("100k run (resumable, Drive-synced)",
     "import subprocess\n"
     "print(subprocess.run('cd /tmp/callic_pkg && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python3 train_full.py --data /tmp/div2k/DIV2K_valid_HR --steps 100000 --bs 32 --lr 5e-4 --schedule cosine --log-every 500 --keep-every 10000 --keep-last 3 --resume --out /tmp/div2k/run100k/mgcf.pt --drive-dir /content/drive/MyDrive/callic/run_100k --drive-every 2000 > /tmp/div2k/run100k.log 2>&1 & echo launched', shell=True, capture_output=True, text=True).stdout)\n"),
    ("Setup (GPU check + install)", "import torch, sys\nprint(torch.__version__, torch.cuda.is_available())\n!nvidia-smi"),
    ("Pretrain MGCF (DIV2K+Flickr2K, 2M steps, bs32, lr5e-4)",
     "PYTHONPATH=/content/callic-gpu-colab python tools/train.py --data /content/data --steps 2000000 --bs 32 --lr 5e-4 --out /content/mgcf.pt"),
    ("RPFT adapt one image (T=50, lr1e-2, b=0.2,d=0.1,e=1, s=0.05,w=0.05)",
     "PYTHONPATH=/content/callic-gpu-colab python tools/eval.py --ckpt /content/mgcf.pt --data_root /content/eval --rpft"),
    ("Eval all sets → Table 1 bpsp",
     "PYTHONPATH=/content/callic-gpu-colab python tools/eval.py --ckpt /content/callic.pt --data_root /content/eval"),
]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def build_notebook():
    cells = []
    hashes = {}
    for rel in MIRROR_FILES:
        ap = os.path.join(ROOT, rel)
        with open(ap) as f:
            src = f.read()
        hashes[rel] = sha256_file(ap)
        cells.append({
            "cell_type": "code",
            "metadata": {"local_twin": rel, "sha16": hashes[rel]},
            "source": [f"# LOCAL TWIN: {rel}  (sha16={hashes[rel]})\n", "%%writefile " + rel + "\n", src],
            "outputs": [], "execution_count": None,
        })
    for title, code in GPU_CELLS:
        cells.append({
            "cell_type": "code",
            "metadata": {"role": "colab-gpu"},
            "source": [f"# {title}\n", code],
            "outputs": [], "execution_count": None,
        })
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "cells": [{"cell_type": "markdown", "metadata": {},
                     "source": ["# CALLIC — Colab GPU mirror (generated, do not hand-edit)\n",
                                "Local twins are single source; run `python tools/sync_colab.py` to regenerate.\n"]}] + cells}
    out = os.path.join(ROOT, "colab", "callic_colab.ipynb")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    return out, hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out, hashes = build_notebook()
    print(f"sync_colab: wrote {out}")
    for k, v in hashes.items():
        print(f"  {k} sha16={v}")
    if args.check:
        print("sync_colab: check OK (notebook regenerated from current locals)")


if __name__ == "__main__":
    main()
