# Autoresearch: CALLIC faithful reproduction (no overfit, no cheating)

## Objective

Implement CALLIC (arXiv:2412.17464v1) from scratch in PyTorch so measured bpsp matches
paper Table 1 with honest evaluation: MGCF Kodak 2.77 / RS19 1.94 / Histo24 2.88 /
DIV2K 2.49 / CLIC.p 2.33; CALLIC Kodak 2.54 / RS19 1.74 / Histo24 2.74 / DIV2K 2.46 /
CLIC.p 2.30. Hold to paper details: MCG Eq.5, MGCF Fig.1b (depth 3, dim 128, k7),
CCI Fig.1c (cache-then-crop, 3P-2 grouped steps, Type-A/B), LoRA Eq.6 + Tucker Eq.7
(~25K mergeable), RPFT schedule Eq.8 (b=0.2,d=0.1,e=1,T=50,lr=1e-2), MDL loss Eq.9
(s=0.05,w=0.05). Heavy training/eval runs on Colab GPU via colab-mcp; every Colab
cell is mirrored to a local file so all edits happen locally and are copied to Colab.

## Metrics

- **Primary**: kodak_bpsp (bits per sub-pixel, lower is better) — honest NLL-based entropy incl. weight bits.
- **Secondary**: rs19_bpsp, histo24_bpsp, div2k_bpsp, clicp_bpsp, params, mergeable_params, enc_time_s.

## How to Run

`./.auto/measure.sh` — outputs `METRIC name=value` lines. Fast pre-check (<1s syntax/import),
then tiny honest smoke eval (2 synthetic images + CCI parity + round-trip), no test-set training.
Full-dataset numbers come from Colab runs recorded in `.auto/log.jsonl` ASI, never hard-coded.

## Files in Scope

- `callic/mcg.py` — Masked Convolutional Gating (1x1 WA/WV, kxk masked DWConv, swish gate, 1x1 out).
- `callic/mgcf.py` — embedding 3x3 Type-B + 3x (LN→MCG→resid + LN→MLP(GELU,x4)→resid) + 1x1 mixture head + masks.
- `callic/cci.py` — Cache-then-Crop grouped autoregressive inference, patch-parallel 3P-2 steps.
- `callic/adapt.py` — LoRA + Tucker DWConv, STE quant w=0.05, logistic prior s=0.05, MDL loss Eq.9.
- `callic/rpft.py` — descending-rate patch sorting, F(t) smoothstep schedule, T=50 fine-tune.
- `callic/mixture.py` — discrete logistic mixture (K=10) NLL.
- `callic/coder.py` — entropy bookkeeping + range/arithmetic coder hookup (NLL first, real bits later).
- `tools/train.py`, `tools/eval.py`, `tools/sync_colab.py` — pretrain (DIV2K+Flickr2K 64x64, Adam 2M/bs32/lr5e-4), eval, Colab mirror.
- `tests/` — mask causality, CCI≡naive parity, merge equality, round-trip.

## Off Limits

- Do NOT hard-code bpsp tables or tune on test images outside per-image RPFT with weight-bits counted.
- Do NOT copy pretrained weights from other codecs; train or init from scratch with fixed seeds.
- Do NOT modify `2412.17464v1.pdf`, `.auto/log.jsonl` by hand, or commit Colab credentials.

## Constraints

- Tests in `.auto/checks.sh` must pass (no mask leakage, CCI parity, lossless round-trip, param budgets 570–580K / 23–27K mergeable).
- No new heavy deps beyond torch/numpy/pillow; keep CPU-smoke fast, GPU-full in Colab.
- Every Colab cell must have a local file twin; log cell IDs + file hashes in ASI.

## What's Been Tried

- (init) Paper visually verified pp.1–10; spec at docs/superpowers/specs/2026-09-11-callic-design.md (MLP x4, K=10, P=64 inferred, asserted in code).
- (init) Colab MCP connected (browser connection open, 8 tools); local GTX 1650 Ti + torch CPU-only, so full runs go to Colab.
