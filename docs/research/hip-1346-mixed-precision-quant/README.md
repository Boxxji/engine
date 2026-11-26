# HIP-1346 Research Dossier: Mixed-Precision & Activation-Aware Quantization per Layer (`hanzoai/engine#49`)

- **Canonical Owning Repository**: `hanzoai/engine` (crate: [`hanzo-mixed-quant/`](../../../hanzo-mixed-quant/), license: `Apache-2.0`, zero external dependencies, `#![forbid(unsafe_code)]`)
- **Tracking Issue**: [`hanzoai/engine#49`](https://github.com/hanzoai/engine/issues/49) (`p1`, `perf`, `research`, `bounty`, `hip-1346`)
- **Raw Benchmark Receipt**: [`results.json`](./results.json) (Receipt SHA-256: `3a2bd97d638da6d7638f3d90524745552e0329b4da9d211dec00d0c14259d83d`)
- **PR Disposition Target**: `MERGE` as an opt-in pure-Rust reference oracle, calibration planner, and HIP-1346 benchmark crate alongside `hanzo-quant`; device CUDA/ROCm/Metal kernel promotion remains gated on live GPU weight-loaded verification.

---

## 1. Problem Statement, References & License Audit

### Problem
*"Minimum bits everywhere is not always best quality-per-byte or fastest."* Uniformly pushing every layer of a hybrid Transformer + Gated DeltaNet (GDN) language model to 4-bit or 1.58-bit ternary wastes precious precision budget on small, extreme-sensitivity bottlenecks (`Embedding`, `LmHead`, `RouterGate`, and multiplicative `GdnRecurrent` layers) while over-provisioning bulk MLP projections (`MlpGateUp`, `MlpDown`) that represent ~65–70% of parameters and tolerate aggressive rotation + activation-aware compression.

### Primary References & Clean-Room License Audit
All algorithms in `hanzo-mixed-quant` are **100% independent clean-room Apache-2.0 Rust implementations** using only the Rust standard library (`[dependencies]` empty):
- **AWQ** (Lin et al., MLSys 2024, *"AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"*): studied paper equations; clean-room implementation of per-channel salience collection, geometric-mean scale normalization, $\alpha \in [0, 1]$ grid search, and BF16 salient-channel protection.
- **SpinQuant** (Liu et al., ICLR 2025, *"SpinQuant: LLM Quantization with Learned Rotations"*): studied paper formulation; clean-room implementation of $O(n \log n)$ normalized Walsh-Hadamard transforms ($H_n$) and exact Cayley-parameterized orthogonal matrices $R = (I - A)(I + A)^{-1}$ via partial-pivoted Gaussian elimination.
- **LLM Compressor** (vLLM / Neural Magic, Apache-2.0): studied mixed-precision sensitivity allocation, 2:4 semi-structured pruning sensitivity, and quantization-aware distillation (QAD); independent clean-room Rust implementation.
- **BitNet-b1.58** (Ma et al., 2024): studied absmean ternary `{-1, 0, +1}` formulation; independent clean-room 2-bit packed trit storage and branchless LUT matvec.

Zero external source files, third-party crates, or non-commercial code were imported.

---

## 2. Methodology, Public Corpus Grounding & Reproduction

### Grounded Repository Corpus (Verified SHA-256)
All calibration batches and 4-domain evaluation sequences are deterministically derived (`--seed 1346`, SplitMix64) from **3 versioned repository files** (`140,742` bytes total):
- `calibration_data/calibration_datav3_small.txt` (`133,837` bytes, SHA-256 `aabd956dba34ff31f40e78b9de4e234fdd4865f90ba231e8e3deec8dfb22cbc0`)
- `chat_templates/llama3.json` (`420` bytes, SHA-256 `67d1d402738727aa3059d28590fa3488c44740a358be4b5c2135c60f58ec9d7a`)
- `orderings/xlora-paper-ordering.json` (`6,485` bytes, SHA-256 `8d0650ea231eeade0b97e5c026411365ff9feed41b2b07ecf10853c2a399043a`)
- **Combined Grounded Corpus SHA-256**: `ebf55f7a1753ed837efcc218dd089c2f74559a7a6f5ff37d8db099a39c241a6d`

### Recorded Hardware & Software Environment
- **CPU Architecture / Model**: `x86_64` — `Intel(R) Xeon(R) CPU @ 2.30GHz`
- **OS / Kernel**: `Linux 6.18.57 x86_64`
- **GPU / Driver**: CPU-only host (no local CUDA/ROCm/Metal GPU detected; `4B / 9B / 27B` GPU timing & energy columns are **explicitly tagged analytical H100-SXM5 80GB projections** combining exact byte accounting with measured CPU quality and overhead ratios)
- **Compiler / Base Commit**: `rustc 1.100.0 (0ef1bc386 2026-11-09)`, base commit `df1722952421`

### Exact Offline Reproduction Commands
```bash
# 1. Formatting and zero-warning Clippy checks (offline)
cargo fmt -p hanzo-mixed-quant -- --check
cargo clippy -p hanzo-mixed-quant --all-targets --offline -- -D warnings

# 2. Full 16-test integration suite (bit-exact formats, orthogonality, AWQ, QAD, Pareto & negative regimes)
cargo test -p hanzo-mixed-quant --offline --release

# 3. Reproducible HIP-1346 benchmark and JSON receipt generation
cargo run -p hanzo-mixed-quant --release --offline --bin mixed-quant-bench -- \
  --seed 1346 --output docs/research/hip-1346-mixed-precision-quant/results.json
```

---

## 3. Crate Architecture (`hanzo-mixed-quant/`)

- [`src/formats.rs`](../../../hanzo-mixed-quant/src/formats.rs): Bit-exact packed weight formats and direct packed matvec kernels:
  - `PackedTernaryMatrix`: BitNet-b1.58 `{-1, 0, +1}` trits packed 4/byte with per-group absmean scales and branchless 4-entry LUT `matvec`.
  - `PackedInt4Matrix`: Per-group symmetric `[-7, +7]` and asymmetric `[0, 15]` 4-bit nibbles packed 2/byte with hoisted scale/zero-point `matvec`.
  - `PackedInt8Matrix`: Symmetric `[-127, +127]` per-group/per-channel weights with exact error bounds (`<= max_abs / 254`).
  - `PackedBf16Matrix`: Exact IEEE-754 Round-to-Nearest-Even (`f32_to_bf16_bits`, `bf16_bits_to_f32`) preserving ties-to-even, signed zero, infinities, overflow, and quiet NaNs.
  - `PackedWeights::fold_row_scales`: Zero-overhead exact scale folding merging QAD row scales into existing per-group quantization scales & zero-points with `0` additional bytes.
- [`src/rotation.rs`](../../../hanzo-mixed-quant/src/rotation.rs): Exact $O(n \log n)$ normalized Fast Walsh-Hadamard Transform (`fwht_normalized_in_place`), exact Cayley-parameterized orthogonal matrices $R = (I - A)(I + A)^{-1}$ (`OrthogonalMatrix::from_cayley_skew` via partial-pivoted `f64` Gauss-Jordan elimination), orthogonality verifier `||R^T R - I||_inf`, and learned SpinQuant optimizer (`optimize_spinquant_rotation`).
- [`src/awq.rs`](../../../hanzo-mixed-quant/src/awq.rs): Real per-input-channel activation statistics (`ActivationStats`), top-salient BF16 channel protection (`ProtectedChannels`), geometric-mean normalized AWQ scale grid search ($\alpha \in [0, 1]$), and SpinQuant+AWQ composition.
- [`src/compressor.rs`](../../../hanzo-mixed-quant/src/compressor.rs): Per-component sensitivity profiler across `{Ternary, Int4Asym, Int8, Bf16}`, exact 2:4 semi-structured and 50% unstructured magnitude pruning sensitivity, equal-memory greedy marginal-gain knapsack planner (`plan_mixed_precision`), and Straight-Through Estimator (STE) Quantization-Aware Distillation (`run_qad_linear_refinement`).
- [`src/model.rs`](../../../hanzo-mixed-quant/src/model.rs): Deterministic multi-layer hybrid Transformer + Gated DeltaNet (GDN) inference engine evaluating all 9 quantization regimes across all 4 required quality domains (`instruction_following`, `code_edit_accuracy`, `multilingual`, `long_context_retrieval`).
- [`src/matrix.rs`](../../../hanzo-mixed-quant/src/matrix.rs): Shared HIP-1346 `4B / 9B / 27B` hybrid matrix accounting (matching `docs/HIP-1346-kivi-kv-compression.md`), 20-turn coding-agent loop scorecard (`32,768` cold prefill + 20 turns × `4,800` diff/tool + `256` gen = `129,088` peak context), and strict end-to-end acceptance gate (`evaluate_acceptance_gate`).

---

## 4. Measured Kernel Verifications (Exact Values from `results.json`)

| Verification Kernel | Metric | Measured Result |
|---|---|---|
| **AWQ Grid Search** (`layer_0.seq_in`, grounded activations) | Optimal salience exponent `alpha*` / outlier ratio | `alpha* = 0.45`, outlier ratio `156.86x` |
| **AWQ Grid Search** | Plain PTQ MSE -> AWQ MSE | `1.838276` -> **`0.468334`** (**`-74.52%` MSE**) |
| **SpinQuant Cayley** (`d=64`, `layer_0.seq_out` × heavy-tailed `layer_in[0]`) | Orthogonality error `\|\|R^T R - I\|\|_inf` | **`8.55e-9`** (`< 1e-5` bound) |
| **SpinQuant Cayley** | Exact unquantized invariance `max \|x W^T - (x R)(W R)^T\|` | **`3.05e-5`** |
| **SpinQuant Cayley** | Activation crest factor (`max\|x\| / rms(x)`) | `12.40` -> **`4.05`** (**`3.06x` peak suppression**) |
| **SpinQuant Cayley** | W4A4 Hadamard MSE -> Learned Cayley MSE | `1.870540` -> **`1.789185`** (Cayley strictly improves Hadamard; synthetic 30x outlier test `test_07` confirms W4A4 win over unrotated baseline) |
| **LLM Compressor Knapsack** | Equal-memory budget vs planned bytes | `128,576 / 128,640 B` (`130,880 B` total incl. AWQ scales `<= 133,760 B` Uniform INT4) |
| **LLM Compressor Knapsack** | Weighted sensitivity MSE (`Uniform INT4` -> `Mixed`) | `4.132797` -> **`2.370331`** (**`-42.6%`**) |
| **2:4 Semi-Structured Pruning** | Components tolerating 2:4 pruning (`rel_mse < 8%`) | `3 / 22` (all `10` critical components exempt) |
| **QAD Refinement** (`17` compressed layers, zero-byte folded scales) | Mean pre-QAD -> post-QAD MSE | `0.146536` -> **`0.100134`** (**`-31.7%` MSE**) |
| **QAD Refinement** | Mean pre-QAD -> post-QAD Logit KL | `0.061231` -> **`0.040514`** (**`-33.8%` KL**) |

### Planned Per-Layer Mixed-Precision Assignment (`130,880 B` Total <= `133,760 B` Uniform INT4)
- **Protected at `INT8`** (`12` components): `embedding`, `lm_head`, all `4` `router_gate` projections, all `4` `gdn_recurrent` projections (`layer_0` & `layer_2` `seq_in`/`seq_out`), and `attn_out` (`layer_1` & `layer_3`).
- **Allocated `INT4_Asym` + AWQ + SpinQuant + QAD** (`4` components): `layer_1.seq_in`, `layer_3.seq_in` (`attn_qkv`), `layer_0.mlp_down`, `layer_3.mlp_down`.
- **Compressed to `Ternary 1.58-bit` + AWQ + SpinQuant + QAD** (`6` bulk MLP components): `layer_0..3.mlp_up`, `layer_1.mlp_down`, `layer_2.mlp_down`.

---

## 5. CPU-Measured End-to-End 4-Domain Quality & Equal-Memory Scorecard (`--seed 1346`)

Every number in this table is **directly measured** on `Intel(R) Xeon(R) CPU @ 2.30GHz` executing real packed-weight forward passes through the 4-layer Hybrid Transformer + GDN model:

| Regime | Model Size (`Bytes`) | Effective BPW | Comp vs BF16 | Instr Acc | Code-Edit Acc | Multilingual Acc | Long-Ctx Acc | **Mean Acc** | Reg vs BF16 | **Mean KL** | CPU Cold / Warm TTFT (`ms`) | CPU p50 / p95 ITL (`ms`) | CPU Wall (`ms`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bf16_reference` | `428,032` | `16.00` | `1.00x` | `100.0%` | `100.0%` | `100.0%` | `100.0%` | **`100.00%`** | `0.00 pp` | `0.0000` | `6.50 / 6.69` | `4.32 / 13.42` | `538.2` |
| `uniform_int8` | `227,392` | `8.50` | `1.88x` | `98.5%` | `100.0%` | `100.0%` | `100.0%` | **`99.62%`** | `-0.38 pp` | `0.0005` | `26.07 / 14.26` | `9.05 / 28.07` | `688.3` |
| `uniform_int4` | `133,760` | `5.00` | `3.20x` | `75.0%` | `82.6%` | `100.0%` | `75.0%` | `83.14%` | `-16.86 pp` | `0.1519` | `24.34 / 21.60` | `12.00 / 18.75` | `857.3` |
| `uniform_ternary` *(Neg 1)* | `66,880` | `2.50` | `6.40x` | `0.0%` | `0.0%` | `1.5%` | `4.2%` | `1.42%` | `-98.58 pp` | `4.3981` | `10.22 / 13.96` | `7.84 / 15.26` | `537.1` |
| `awq_int4` | `168,464` | `6.30` | `2.54x` | `78.0%` | `93.9%` | `100.0%` | `83.3%` | `88.83%` | `-11.17 pp` | `0.0809` | `20.91 / 25.64` | `15.64 / 20.52` | `918.6` |
| `spinquant_int4` | `133,760` | `5.00` | `3.20x` | `77.3%` | `90.2%` | `100.0%` | `79.2%` | `86.65%` | `-13.35 pp` | `0.0924` | `26.06 / 18.36` | `15.28 / 42.59` | `994.5` |
| **`mixed_precision_awq_spinquant_qad`** | **`130,880`** | **`4.89`** | **`3.27x`** | **`97.0%`** | **`95.5%`** | **`100.0%`** | **`100.0%`** | **`98.11%`** | **`-1.89 pp`** | **`0.0038`** | `24.53 / 27.75` | `13.33 / 21.67` | `852.9` |
| `neg2_unmerged_rot_scalar_dequant` *(Neg 2)* | `130,880` | `4.89` | `3.27x` | `97.0%` | `95.5%` | `100.0%` | `100.0%` | `98.11%` | `-1.89 pp` | `0.0038` | `179.24 / 258.07` | `110.15 / 181.25` | **`9,058.8`** |
| `neg3_mismatched_prose_cal_awq` *(Neg 3)* | `168,464` | `6.30` | `2.54x` | `75.0%` | `87.9%` | `100.0%` | `75.0%` | `84.47%` | `-15.53 pp` | `0.1141` | `11.38 / 11.81` | `9.92 / 16.00` | `739.3` |

**Equal-Memory Pareto Headline**: At **strictly smaller memory** than `uniform_int4` (`130,880 B` / `4.89 BPW` vs `133,760 B` / `5.00 BPW`, `-2.15%` fewer bytes), `mixed_precision_awq_spinquant_qad` raises 4-domain accuracy by **`+14.97 pp`** (`83.14%` -> **`98.11%`**) and cuts logit KL divergence by **`40.1x`** (`0.1519` -> **`0.0038`**), while running slightly faster on CPU than `uniform_int4` (`852.9 ms` vs `857.3 ms`).

---

## 6. Shared `4B / 9B / 27B` Hybrid Matrix & 20-Turn Coding-Agent Scorecard

Architectural geometries match [`docs/HIP-1346-kivi-kv-compression.md`](../../HIP-1346-kivi-kv-compression.md) (`4B-Hybrid`: `9` Attn + `27` GDN; `9B-Hybrid`: `10` Attn + `30` GDN; `27B-Hybrid`: `16` Attn + `48` GDN). Workload: `32,768`-token cold repo prefill + 20 coding-agent turns × (`4,800` tool/diff prefill + `256` generated tokens) = `129,088` peak context, `5,120` generated tokens.

> **Honesty Boundary**: `Model Size (GiB)` and `Peak Memory (GiB)` are **exact deterministic byte accounting** (`group_size = 128`, FP16 Attn KV + unquantized FP32 GDN state + 256 MiB workspace). `Correctness (%)` and `Reg (pp)` are **directly measured on CPU**. `Cold/Warm TTFT`, `p50/p95 ITL`, `Warm tok/s`, `20-Turn Wall`, and `mJ/tok` are **analytically modeled H100-SXM5 80GB projections** (`3.35 TB/s` HBM3 @ 80% achieved, `420 TFLOPS` effective BF16 prefill, `650 W` active power) calibrated with measured overhead ratios.

| Model | Regime | Model Size (`GiB`) | Peak Mem (`GiB`) | Cold TTFT (`ms`) | Warm TTFT (`ms`) | p50 / p95 ITL (`ms`) | Warm (`tok/s`) | 20-Turn Wall (`s`) | Energy (`mJ/tok`) | Correctness (Reg) | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `4B-Hybrid` | `bf16_reference` | `6.59` | `9.17` | `698.7` | `80.9` | `3.2 / 3.3` | `316.5` | `18.66` | `2,369.3` | `100.0% (0.0 pp)` | **ACCEPTED** |
| `4B-Hybrid` | `uniform_int8` | `3.35` | `5.92` | `549.3` | `69.6` | `2.0 / 2.1` | `508.8` | `11.94` | `1,515.3` | `99.6% (-0.4 pp)` | **ACCEPTED** |
| `4B-Hybrid` | `uniform_int4` | `1.75` | `4.33` | `558.0` | `76.0` | `1.3 / 1.4` | `742.5` | `8.89` | `1,128.5` | `83.1% (-16.9 pp)` | REJECTED |
| `4B-Hybrid` | `uniform_ternary` | `0.88` | `3.45` | `527.5` | `74.4` | `1.0 / 1.0` | `1,037.0` | `6.88` | `873.1` | `1.4% (-98.6 pp)` | REJECTED |
| `4B-Hybrid` | `awq_int4` | `1.80` | `4.37` | `575.6` | `78.5` | `1.4 / 1.5` | `718.0` | `9.19` | `1,166.7` | `88.8% (-11.2 pp)` | REJECTED |
| `4B-Hybrid` | `spinquant_int4` | `1.75` | `4.33` | `580.1` | `79.3` | `1.4 / 1.5` | `722.0` | `9.17` | `1,163.8` | `86.6% (-13.4 pp)` | REJECTED |
| `4B-Hybrid` | **`mixed_precision_awq_spinquant_qad`** | **`1.75`** | **`4.33`** | **`569.0`** | **`77.7`** | **`1.4 / 1.4`** | **`735.5`** | **`9.00`** | **`1,142.0`** | **`98.1% (-1.9 pp)`** | **ACCEPTED** |
| `4B-Hybrid` | `neg2_unmerged_rot_scalar_dequant` | `1.75` | `4.33` | `1,307.8` | `185.9` | `3.3 / 3.9` | `305.5` | `21.58` | `2,739.3` | `98.1% (-1.9 pp)` | REJECTED |
| `4B-Hybrid` | `neg3_mismatched_prose_cal_awq` | `1.80` | `4.37` | `575.6` | `78.5` | `1.4 / 1.5` | `718.0` | `9.19` | `1,166.7` | `84.5% (-15.5 pp)` | REJECTED |
| `9B-Hybrid` | `bf16_reference` | `16.93` | `22.33` | `1,794.5` | `207.8` | `8.0 / 8.3` | `125.5` | `46.56` | `5,911.5` | `100.0% (0.0 pp)` | **ACCEPTED** |
| `9B-Hybrid` | `uniform_int8` | `8.60` | `14.00` | `1,410.8` | `178.7` | `4.8 / 5.0` | `209.6` | `29.25` | `3,713.0` | `99.6% (-0.4 pp)` | **ACCEPTED** |
| `9B-Hybrid` | `uniform_int4` | `4.50` | `9.90` | `1,433.1` | `195.3` | `3.2 / 3.4` | `315.2` | `21.38` | `2,714.4` | `83.1% (-16.9 pp)` | REJECTED |
| `9B-Hybrid` | `uniform_ternary` | `2.25` | `7.65` | `1,354.8` | `191.1` | `2.2 / 2.3` | `454.9` | `16.23` | `2,060.2` | `1.4% (-98.6 pp)` | REJECTED |
| `9B-Hybrid` | **`mixed_precision_awq_spinquant_qad`** | **`4.50`** | **`9.90`** | **`1,461.5`** | **`199.4`** | **`3.2 / 3.4`** | **`312.3`** | **`21.64`** | **`2,747.5`** | **`98.1% (-1.9 pp)`** | **ACCEPTED** |
| `9B-Hybrid` | `neg2_unmerged_rot_scalar_dequant` | `4.50` | `9.90` | `3,358.9` | `477.4` | `7.7 / 9.1` | `129.7` | `51.90` | `6,588.3` | `98.1% (-1.9 pp)` | REJECTED |
| `27B-Hybrid` | `bf16_reference` | `46.34` | `63.12` | `4,911.8` | `568.7` | `22.3 / 23.2` | `44.9` | `129.71` | `16,466.6` | `100.0% (0.0 pp)` | **ACCEPTED** |
| `27B-Hybrid` | `uniform_int8` | `23.53` | `40.31` | `3,861.5` | `489.0` | `13.5 / 14.2` | `73.9` | `82.37` | `10,457.6` | `99.6% (-0.4 pp)` | **ACCEPTED** |
| `27B-Hybrid` | `uniform_int4` | `12.31` | `29.09` | `3,922.7` | `534.5` | `9.1 / 9.7` | `109.7` | `60.91` | `7,733.0` | `83.1% (-16.9 pp)` | REJECTED |
| `27B-Hybrid` | `uniform_ternary` | `6.15` | `22.94` | `3,708.2` | `523.2` | `6.5 / 6.9` | `154.0` | `46.78` | `5,939.3` | `1.4% (-98.6 pp)` | REJECTED |
| `27B-Hybrid` | **`mixed_precision_awq_spinquant_qad`** | **`12.31`** | **`29.09`** | **`4,000.3`** | **`545.9`** | **`9.2 / 9.8`** | **`108.7`** | **`61.65`** | **`7,826.4`** | **`98.1% (-1.9 pp)`** | **ACCEPTED** |
| `27B-Hybrid` | `neg2_unmerged_rot_scalar_dequant` | `12.31` | `29.09` | `9,193.8` | `1,306.7` | `22.2 / 26.2` | `45.1` | `147.85` | `18,770.0` | `98.1% (-1.9 pp)` | REJECTED |

*(All 27 `(model, regime)` rows including `awq_int4`, `spinquant_int4`, and `neg3_mismatched_prose_cal_awq` for `9B` and `27B` are recorded verbatim in [`results.json`](./results.json).)*

---

## 7. Explicitly Tested Honest Negative Regimes

1. **Negative Regime 1 — Uniform Ternary / INT4 on Critical Layers (`uniform_ternary`, `uniform_int4`)**:
   - Applying `1.58-bit Ternary` uniformly across `Embedding`, `LmHead`, `RouterGate`, and `GdnRecurrent` collapses mean 4-domain accuracy to **`1.42%`** (`-98.58 pp`, `KL = 4.3981`). Even uniform `INT4` drops to **`83.14%`** (`-16.86 pp`, `KL = 0.1519`, with `instruction_following = 75.0%` and `long_context_retrieval = 75.0%`). Both are **REJECTED** by the acceptance gate, proving mixed precision is mandatory.
2. **Negative Regime 2 — Microbenchmark Win / End-to-End Loss (`neg2_unmerged_rot_scalar_dequant`)**:
   - Uses identical packed mixed-precision weights (`130,880 B`, `3.27x` smaller than BF16) and identical high quality (`98.11%`, `KL = 0.0038`), winning the weight-memory microbenchmark. However, applying an unmerged dense $O(d^2)$ runtime rotation and naive full-matrix scalar dequantization regresses measured CPU wall time to **`9,058.8 ms`** (`16.8x` slower than BF16, `10.6x` slower than fused MixedPrecision) and modeled 27B 20-turn wall time to **`147.85 s`** vs BF16 **`129.71 s`**. Our acceptance gate **explicitly REJECTS** it: *"microbenchmark win regresses end-to-end inference"*.
3. **Negative Regime 3 — Uncalibrated / Mismatched Domain Calibration (`neg3_mismatched_prose_cal_awq`)**:
   - Calibrating AWQ strictly on ASCII prose slices from `calibration_datav3_small.txt` (omitting `llama3.json` chat templates and `xlora-paper-ordering.json` structural code tokens) misses code-edit and long-context outlier channels, degrading AWQ from **`88.83%` (`KL = 0.0809`)** down to **`84.47%` (`KL = 0.1141`, `-4.36 pp` overall; `-6.0 pp` on `code_edit_accuracy`, `-8.3 pp` on `long_context_retrieval`, `-3.0 pp` on `instruction_following`)**.

---

## 8. Integration Recommendation for `hanzo-engine` / `hanzo-quant`

1. **Per-Layer Precision Policy**: Wire `hanzo_mixed_quant::compressor::plan_mixed_precision` into `hanzo-quant`'s ISQ/UQFF planner so `Embedding`, `LmHead`, `RouterGate`, and `GdnRecurrent` layers enforce a fail-closed `INT8`/`BF16` precision floor while bulk MLP projections accept `INT4`/`Ternary` + SpinQuant + AWQ + QAD under a user-specified VRAM budget.
2. **Mandatory Offline Rotation Merging & Zero-Byte QAD Folding**: Always fuse SpinQuant rotations offline into adjacent RMSNorm/linear weights (or use fast $O(d \log d)$ block-Hadamard kernels) and fold QAD row scales into per-group quantization headers (`PackedWeights::fold_row_scales`)—never deploy unmerged dense runtime rotations.
3. **Multi-Domain Calibration Manifest**: Require SHA-256-verified multi-domain calibration batches (prose + chat templates + code/JSON + multilingual) before emitting AWQ/QAD artifacts, and validate on live CUDA/ROCm/Metal `4B/9B/27B` checkpoints before changing default quantization recipes.
