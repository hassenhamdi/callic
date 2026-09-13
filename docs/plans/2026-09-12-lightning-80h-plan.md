# Lightning AI 80h — CALLIC full-recipe plan (2026-09-12)

Goal: paper-exact MGCF pretrain (2M steps, DIV2K+Flickr2K) + full Table 1 benchmark + RPFT/CALLIC numbers, inside 80 GPU-hours with margin.

## 0. GPU pick (strongest the credits allow, in this order)

| GPU | ~patches/s (bs32) | 2M steps (64M patches) | Verdict |
|---|---|---|---|
| A100 40GB | ~2000 | **~9h** | Best: full recipe + ablations + RPFT headroom |
| L4 / A10G | ~700 | **~25h** | Best value: full recipe comfortably |
| T4 | 400 (measured) | **~44h** | Works: full recipe + evals, tight but fits |

Rule: 80h must cover ≥2× the recipe column. T4 qualifies; anything stronger buys ablations.

## 1. Phase 0 — setup (~1h, CPU ok): `bash tools/lightning_setup.sh`
Downloads DIV2K train/val (official, verified), Flickr2K (HF mirror), Kodak 24. Verifies code compiles + checks pass.

## 2. Phase 1 — optimizer shootout (~1–3h, the only experiment before committing)
Same data, 3k steps each, bs32, compare loss-vs-step + wall time:
- (a) AdamW 5e-4 (paper baseline)
- (b) Muon hybrid (Muon on ≥2D, AdamW on biases/norms/head, lr≈0.02 + warmup)
- (c) Aurora/NorMuon row-norm on tall `W_up` only (if (b) implemented)

Gate: adopt winner for Phase 2 only on clear sample-efficiency win. (Status: `--opt` harness not yet implemented — say the word and it's next.)

## 3. Phase 2 — full pretrain (the bulk: 9–44h by GPU)
Paper recipe with winner config + `cudnn.benchmark`, best+numbered ckpts, resume-safe:
```
python tools/train.py --data data --steps 2000000 --bs 32 --lr 5e-4 \
  --schedule cosine --log-every 500 --keep-every 50000 --keep-last 3 --resume \
  --out checkpoints/mgcf_full.pt
```
Keep-every 50k (7MB each — cheap). Mid-run gate at ~200k steps: loss should be ≤ ~4 (our T4 run hit 4.8 @200 on 10× less data).

## 4. Phase 3 — full benchmark (~2–6h)
- MGCF rows: Kodak 24, DIV2K-val 100, CLIC.p, Histo24, RS19-190 (targets 2.77/2.49/2.33/2.88/1.94).
- CALLIC rows: per-image RPFT T=50 + weight bits (targets 2.54/2.46/2.30/2.74/1.74).
- Cost warning: RPFT T=50 ≈ 60s/image on T4 → RS19-190 alone ≈ 3h. Use T=10 fast mode (paper Fig. 3: 3.2s, 2.62) for large sets or subsample.
- Missing-data flags: RS19/Histo24/CLIC need manual prep (not in setup script); Kodak + DIV2K-val are ready.

## 5. Phase 4 — ablations with leftover hours (Table 3 / Fig. 3–4 style)
Depth/dim/kernel variants (short 50k-step runs), T-vs-bpsp sweep, RPFT config sweep. Only if Phases 2–3 are banked.

## Budget sketch (A100 case)
Setup 1h + shootout 2h + pretrain 9–12h + benchmark 3h + ablations 10h ≈ 28h — under half the budget, rest is insurance for restarts and quota gaps. On T4: 1 + 3 + 44 + 5 ≈ 53h — fits, no ablations.
