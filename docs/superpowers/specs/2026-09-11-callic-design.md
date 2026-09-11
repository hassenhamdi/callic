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
- **Params:** target 575K MGCF; CALLIC adds ~25K mergeable (see §3). Fail build if outside 570–580K / 23–27K.

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
- **Quant:** step `w=0.05 (<1)`; STE for inference `φ̂=sg(⌊φ/w⌉·w−φ)+φ`; uniform noise `φ̃=φ+U(−w/2,w/2)` for rate estimate.
- **Prior:** static logistic zero-mean scale `s=0.05` on `φ̃`.
- **Loss (Eq.9):** `L=−log p_s(φ̃)+Σ_i −log q(x_Gi|x_G<i;θ,φ̂)`.
- **Schedule (Eq.8):** `t'=t/(T(1−d))`, smoothstep `s(x)`, `F(t)=b+(1−b)s(t')^e`, defaults `b=0.2,d=0.1,e=1,T=50,lr=1e-2`.
  Estimate per-patch bpsp, sort descending, train on top-F(t) fraction; final d% steps use full set.

## 4. Training / Eval / Anti-overfit

- **Pretrain:** DIV2K 800 + Flickr2K 2650 → non-overlap `64×64` (≈612,806 patches), Adam 2M steps, bs32, lr5e-4. Colab GPU; mirror code locally.
- **Adapt:** per-test-image RPFT T=50; count weight bits + pixel bits in bpsp.
- **Eval sets:** Kodak 24; RS19 190 center-cropped `576×576`; Histo24 24 `768×512`; DIV2K val; CLIC.p pro val.
- **Metric:** bpsp = total bits / (H·W·3). Targets: MGCF 2.77/1.94/2.88/2.49/2.33; CALLIC 2.54/1.74/2.74/2.46/2.30.
- **Anti-cheat:** train only on train split; no test-tuned hyperparams; no hard-coded tables; entropy bpsp reported separately from real coder bits; seeds fixed; weight rate always included.

## 5. Colab-local sync + testing

- Layout: `callic/{mcg.py,mgcf.py,cci.py,ЗИlora_tucker.py,rpft.py,mixture.py,coder.py}` + `tools/train.py,e eval.py` + Colab cells mirroring modules (single source copied both ways).
- Tests: mask causality; MCG shapes; Tucker/LoRA merge equality; CCI≡naive parity; logistic mixture NLL sanity; lossless round-trip; bpsp smoke on 2 Kodak images; param-count asserts.
- Runtime instrumentation: enc/dec time per Table 2; Fig.3 T-vs-bpsp sweep support.

## Ambiguities made explicit

- MLP expansion 4× and K=10 mixtures inferred from 575K budget — asserted in code, adjustable if count mismatches.
- DLPR scan grouping replicated from description + Fig.1c; parity test guards fidelity.
- Patch size P for coding = 64 default (matches train patch); configurable.
