#!/bin/bash
# CALLIC Lightning AI setup: env + full paper data. Idempotent — safe to re-run.
# Run once per Studio (CPU ok for downloads):  bash tools/lightning_setup.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== 0. GPU check ==="
python3 -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "(no nvidia-smi)"

echo "=== 1. Python deps ==="
python3 -m pip install -q --upgrade pip
python3 -m pip install -q pillow numpy datasets huggingface_hub
echo "=== 1b. Official optimizer code (benchmark harness) ==="
python3 -m pip install -q "muon @ git+https://github.com/KellerJordan/Muon" 2>/dev/null || echo "(muon install skipped)"
python3 -m pip install -q "git+https://github.com/zichongli5/NorMuon.git" 2>/dev/null || echo "(normuon install skipped)"
if [ ! -d thirdparty/aurora-release ]; then
  git clone -q https://github.com/tilde-research/aurora-release.git thirdparty/aurora-release 2>/dev/null || echo "(aurora clone skipped)"
fi

echo "=== 2. Train data: DIV2K (official, verified) ==="
mkdir -p data/DIV2K_train_HR data/DIV2K_valid_HR
if [ "$(ls data/DIV2K_train_HR/*.png 2>/dev/null | wc -l)" -lt 800 ]; then
  curl -L -o /tmp/DIV2K_train_HR.zip https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
  unzip -q -o /tmp/DIV2K_train_HR.zip -d /tmp/div2k_train && mv /tmp/div2k_train/DIV2K_train_HR/*.png data/DIV2K_train_HR/ && rm /tmp/DIV2K_train_HR.zip
fi
if [ "$(ls data/DIV2K_valid_HR/*.png 2>/dev/null | wc -l)" -lt 100 ]; then
  curl -L -o /tmp/DIV2K_valid_HR.zip https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
  unzip -q -o /tmp/DIV2K_valid_HR.zip -d /tmp/div2k_valid && mv /tmp/div2k_valid/DIV2K_valid_HR/*.png data/DIV2K_valid_HR/ && rm /tmp/DIV2K_valid_HR.zip
fi
echo "DIV2K train: $(ls data/DIV2K_train_HR/*.png | wc -l) / valid: $(ls data/DIV2K_valid_HR/*.png | wc -l)"

echo "=== 3. Train data: Flickr2K (HF mirror yangtao9009/Flickr2K) ==="
mkdir -p data/Flickr2K
if [ "$(ls data/Flickr2K/* 2>/dev/null | wc -l)" -lt 2000 ]; then
  python3 - "data/Flickr2K" <<'EOF'
import sys, io
from huggingface_hub import snapshot_download
from datasets import load_dataset
out = sys.argv[1]
try:
    ds = load_dataset("yangtao9009/Flickr2K", split="train")
    import os
    n = 0
    for i, ex in enumerate(ds):
        img = ex["image"] if "image" in ex else ex[list(ex.keys())[0]]
        img.convert("RGB").save(os.path.join(out, f"fk{i:04d}.png"))
        n += 1
    print(f"Flickr2K via datasets: {n}")
except Exception as e:
    print(f"datasets path failed ({str(e)[:150]}), trying snapshot_download")
    p = snapshot_download("yangtao9009/Flickr2K", repo_type="dataset")
    print(f"snapshot at {p} — convert/copy PNGs from there into {out}")
EOF
fi
echo "Flickr2K files: $(ls data/Flickr2K/* 2>/dev/null | wc -l)"

echo "=== 4. Eval data: Kodak 24 (verified mirror) ==="
mkdir -p data/eval/kodak
for i in $(seq -w 1 24); do
  [ -s "data/eval/kodak/kodim${i}.png" ] || curl -sL -o "data/eval/kodak/kodim${i}.png" --max-time 60 \
    "https://raw.githubusercontent.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite/master/PhotoCD_PCD0992/${i}.png"
done
echo "Kodak: $(ls data/eval/kodak/*.png | wc -l)/24"

echo "=== 5. Code checkpoint mirror (Lightning storage persists, still sync) ==="
mkdir -p checkpoints runs
PYTHONPATH=. python3 -m py_compile callic/*.py tools/*.py tests/*.py && echo "compile OK"
bash .auto/checks.sh 2>&1 | tail -n 2
echo "SETUP_DONE — next: see docs/plans/2026-09-12-lightning-80h-plan.md Phase 1"
