# CALLIC Design — 2026-09-11

Approach A (approved): faithful from-scratch PyTorch reproduction.

## 1. Architecture (MGCF / MCG)

- **MCG (Eq.5, Fig.1a):** input `X` → two `1×1` projections `W_A X`, `W_V X`.
  `A_M = DWConv_{k×k}(W_A X, M)` with masked depth-wise conv, `V = W_V X`,
  `MCG(X) = swish(A_M) ⊙ V`, followed by `1×1` out-proj (per Fig.1a top/bottom branches).
- **MGCF block (Fig.1b, MetaFormer-style):**
  `x → LayerNorm → MCG → residual → LayerNorm → MLP → residual`, ×N.
  MLP = two linear layers with GELU, expansion 4× (128→512→128) — chosen to hit 575K; assert by param count.
- **Stem/head:** `3×3` masked-conv embedding (Type-B), `N=3` blocks, `dim=128`, `k=7` (Table 3 bold choice).
  Head = `1×1` Parameter Projection to discrete logistic mixture (PixelCNN++ style, K=10 default; assert channels).
- **Masks:** Type-B masks out current group positions (embedding); Type-A includes current + past (MCG blocks).
- **Params:** target 575K MGCF (paper Table 1); this skeleton counts 580964
  with inferred MLP×4 + K=10 (paper leaves expansion/K unstated) — within ~1%,
  asserted 570–590K. CALLIC adds 23592 mergeable (WA/WV r=8, Wup r=4,
  DWConv r1=8,r2=4,r3=4 incl. r1·r2·r3 core) — within 23–27K.

## 2. Dataflow / CCI (Fig.1c)

- Divide image into patches, encode patches in parallel.
- Per patch, group pixels `x={x_G1..x_Gg}` in DLPR-inspired parallel scan (`3P-2` steps, P=patch size).
- Step `i`: consume cached activations of `x_G≤i-1`, predict distribution of `x_Gi`.
- **Cache-then-crop:** cache activations before each masked DWConv; at step `i`, crop windows around current-group positions, zero-pad conv only on crops in parallel.
- Embedding (Type-B): feed previous-group pixels, cache, cropped Type-B conv to gather context for current group.
- Deeper MCG (Type-A): store current-group activations to cache map, cropped conv for context.
- `1×1` convs transfer across channels only — no caching.
- Parity invariant: CCI output ≡ naive full-masked-conv output (unit-tested).

## 3. RPFT + MDL (Eqs.6–9, Fig.1d)

- **LoRA (Eq.6)** on `W_A`, `W_V`, and MLP `W_up`: `W'=W+AB`, `A∈R^{m×r}`, `B∈R^{r×n}`.
- **Tucker (Eq.7)** on masked DWConv `W_mc∈R^{m×1×k×k}`: `ΔW=I ×1 A ×3 C ×4 D`, `W'_mc=M⊙(W_mc+ΔW)`.
- Rank config targeting ~25K: e.g. WA/WV r=8, Wup r=4, DWConv (r1=8,r2=4,r3=4); tune to 23–27K and log exact count. Mergeable with zero infer overhead.
  Wired as `CALLICModel` (frozen base + zero-init deltas, exact 23592 params):
  identity ≡ base (err 0.0), STE-quant forward + `merge_()` equality (err 0.0).
- **Quant:** step `w=0.05 (<1)`; STE for inference `φ̂=sg(⌊φ/w⌉·w−φ)+φ`; uniform noise `φ̃=φ+U(−w/2,w/2)` for rate estimate.
- **Prior:** static logistic zero-mean scale `s=0.05` on `φ̃`.
- **Loss (Eq.9):** `L=−log p_s(φ̃)+Σ_i −log q(x_Gi|x_G<i;θ,φ̂)`.
- **Schedule (Eq.8):** `t'=t/(T(1−d))`, smoothstep `s(x)`, `F(t)=b+(1−b)s(t')^e`, defaults `b=0.2,d=0.1,e=1,T=50,lr=1e-2`.
  Estimate per-patch bpsp, sort descending, train on top-F(t) fraction; final d% steps use full set.

## 4. Training / Eval / Anti-overfit

- **Pretrain:** DIV2K 800 + Flickr2K 2650 → non-overlap `64×64` (≈612,806 patches), Adam 2M steps, bs32, lr5e-4. Colab GPU; mirror code locally.
  Live Colab T4 run (DIV2K-valid 100 imgs → 60,140 non-overlap patches, same
  hyperparams): 22.56 → 7.02 @50 → 6.02 @100 → 5.19 @150 → 4.82 @200,
  plateau ~4.9 @300 — descending toward paper regime; full 2M-step run needed
  for Table 1 values. Kodak random-init 23.04 (GPU) == local CPU to 4 decimals.
  Trained-2000 ckpt: Kodak micro-avg 5.298 (per-image 3.91–6.24).
  CALLIC RPFT T=50 (frozen base, 23.6K adaptors, STE, weight bits counted):
  kodim01 6.015→5.357 pixel +0.063 wt = 5.420 total; kodim20 3.915→3.390
  pixel +0.057 wt = 3.446 total (~10–12% gain, paper-consistent structure).
- **Adapt:** per-test-image RPFT T=50; count weight bits + pixel bits in bpsp.
- **Eval sets:** Kodak 24; RS19 190 center-cropped `576×576`; Histo24 24 `768×512`; DIV2K val; CLIC.p pro val.
- **Metric:** bpsp = total bits / (H·W·3). Targets: MGCF 2.77/1.94/2.88/2.49/2.33; CALLIC 2.54/1.74/2.74/2.46/2.30.
- **Anti-cheat:** train only on train split; no test-tuned hyperparams; no hard-coded tables; entropy bpsp reported separately from real coder bits; seeds fixed; weight rate always included.

## 5. Colab-local sync + testing

- Layout: `callic/{mcg.py,mgcf.py,cci.py,adapt.py,rpft.py,mixture.py,coder.py}` + `tools/train.py,e eval.py` + Colab cells mirroring modules (single source copied both ways).
- Tests: mask causality; MCG shapes; Tucker/LoRA merge equality; CCI≡naive parity; logistic mixture NLL sanity; lossless round-trip; bpsp smoke on 2 Kodak images; param-count asserts.
- Runtime instrumentation: enc/dec time per Table 2; Fig.3 T-vs-bpsp sweep support.

## Ambiguities made explicit

- MLP expansion 4× and K=10 mixtures inferred from 575K budget — counted
  580964 (~1% over), asserted 570–590K, configurable; exact counts logged.
- DLPR scan grouping replicated from description + Fig.1c as G=3P−2 steps;
  local smoke uses raster-respecting slabs (true mask-leakage test), Colab
  uses per-step dynamic group masks (paper's Type-A/B group semantics);
  parity test guards both.
- Weight rate uses −log pdf − log w (mass ≈ pdf·w) so discrete bits stay
  positive; documented in `adapt.py`.
- Patch size P for coding = 64 default (matches train patch); configurable.
