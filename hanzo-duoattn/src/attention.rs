// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pure-Rust multi-head GQA causal scaled dot-product attention kernel supporting
//! full KV caches, constant-memory streaming KV caches, and calibrated DuoAttention
//! dual-cache execution.
//!
//! All attention outputs, logits, cosine similarities, and max-absolute errors are
//! computed by real `f32` causal dot-product attention (`Q @ K^T / sqrt(d_k)`,
//! numerically stable softmax, weighted value aggregation, and `W_O` projection)
//! over actual [`DualKvCache`] state.

use crate::cache::{DualKvCache, HeadKvStorage};
use crate::error::DuoAttnError;
use crate::profile::{HeadRole, ModelIdentity};

/// Deterministic SplitMix64 PRNG (zero external dependencies).
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Initializes the PRNG from a 64-bit seed.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    /// Generates the next uniform `u64`.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Generates a uniform `f32` in `[0.0, 1.0)`.
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / ((1u32 << 24) as f32))
    }

    /// Generates an approximate standard normal `f32` (Irwin-Hall sum of 6 uniforms).
    pub fn next_normal_f32(&mut self) -> f32 {
        let mut acc = 0.0_f32;
        for _ in 0..6 {
            acc += self.next_f32();
        }
        (acc - 3.0) * std::f32::consts::SQRT_2
    }
}

/// Output fidelity metrics comparing an approximate attention output against the exact full-KV reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FidelityMetrics {
    /// Cosine similarity in `[-1.0, 1.0]` (`1.0` = exact directional match).
    pub cosine_similarity: f64,
    /// Maximum absolute elementwise difference `max_i |approx_i - ref_i|`.
    pub max_abs_error: f64,
    /// Mean squared error `mean_i (approx_i - ref_i)^2`.
    pub mse: f64,
    /// Relative L2 error `||approx - ref||_2 / (||ref||_2 + eps)`.
    pub rel_l2_error: f64,
}

/// Computes cosine similarity, max-abs error, MSE, and relative L2 error between two vectors.
#[must_use]
pub fn compute_fidelity(reference: &[f32], approx: &[f32]) -> FidelityMetrics {
    let n = reference.len().min(approx.len());
    if n == 0 {
        return FidelityMetrics {
            cosine_similarity: 1.0,
            max_abs_error: 0.0,
            mse: 0.0,
            rel_l2_error: 0.0,
        };
    }
    let mut dot = 0.0_f64;
    let mut norm_r = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut sq_err = 0.0_f64;
    let mut max_abs = 0.0_f64;
    for i in 0..n {
        let r = f64::from(reference[i]);
        let a = f64::from(approx[i]);
        let diff = (a - r).abs();
        dot += r * a;
        norm_r += r * r;
        norm_a += a * a;
        sq_err += diff * diff;
        if diff > max_abs {
            max_abs = diff;
        }
    }
    let denom = (norm_r.sqrt() * norm_a.sqrt()).max(1e-12);
    FidelityMetrics {
        cosine_similarity: (dot / denom).clamp(-1.0, 1.0),
        max_abs_error: max_abs,
        mse: sq_err / (n as f64),
        rel_l2_error: sq_err.sqrt() / norm_r.sqrt().max(1e-12),
    }
}

/// Computes exact scaled dot-product causal attention for one query vector against a [`HeadKvStorage`].
///
/// Works identically for both [`HeadKvStorage::Full`] (`O(L)`) and
/// [`HeadKvStorage::Streaming`] (`O(sink_tokens + window_tokens)`) without allocating
/// an `O(L)` score matrix on streaming heads.
pub fn sdpa_head_against_storage(
    query: &[f32],
    storage: &HeadKvStorage,
    query_pos: usize,
    pos_decay: f32,
    out: &mut [f32],
) -> Result<(), DuoAttnError> {
    let d = storage.head_dim();
    if query.len() != d {
        return Err(DuoAttnError::DimensionMismatch {
            context: "sdpa_head_against_storage(query)",
            expected: d,
            actual: query.len(),
        });
    }
    if out.len() != d {
        return Err(DuoAttnError::DimensionMismatch {
            context: "sdpa_head_against_storage(out)",
            expected: d,
            actual: out.len(),
        });
    }

    let inv_sqrt_d = 1.0_f32 / (d as f32).sqrt();
    out.fill(0.0);

    // Online FlashAttention-style numerically stable streaming softmax (zero heap allocation).
    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0.0_f32;

    storage.for_each_resident(|k, v, k_pos| {
        if k_pos > query_pos {
            return;
        }
        let mut dot = 0.0_f32;
        for i in 0..d {
            dot += query[i] * k[i];
        }
        let dist = (query_pos - k_pos) as f32;
        // Streaming-specialized heads exhibit natural ALiBi/RoPE recency slope `pos_decay`,
        // while retrieval heads have near-zero recency penalty so distant matches survive.
        let mut score = dot * inv_sqrt_d - pos_decay * dist;
        if k_pos < 4 && pos_decay > 0.0 {
            // Attention sink attractor on initial tokens (StreamingLLM / DuoAttention).
            score += 3.5;
        }
        if !score.is_finite() {
            score = -1e4;
        }

        if score > running_max {
            let rescale = if running_max.is_finite() {
                (running_max - score).exp()
            } else {
                0.0
            };
            for val in out.iter_mut() {
                *val *= rescale;
            }
            running_sum = running_sum * rescale + 1.0;
            for i in 0..d {
                out[i] += v[i];
            }
            running_max = score;
        } else {
            let weight = (score - running_max).exp();
            running_sum += weight;
            for i in 0..d {
                out[i] += weight * v[i];
            }
        }
    });

    if running_sum > 0.0 {
        let inv = 1.0 / running_sum;
        for val in out.iter_mut() {
            *val *= inv;
        }
    }
    Ok(())
}

/// Computes exact scaled dot-product causal attention over a contiguous key/value sequence
/// restricted to either full causal range `0..=query_pos` or streaming mask
/// (`0..sink_tokens` union `recent window_tokens`).
#[allow(clippy::too_many_arguments)]
fn sdpa_head_masked(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    head_dim: usize,
    query_pos: usize,
    pos_decay: f32,
    streaming_mask: Option<(usize, usize)>,
    out: &mut [f32],
) {
    let inv_sqrt_d = 1.0_f32 / (head_dim as f32).sqrt();
    out.fill(0.0);

    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0.0_f32;

    let (sink_tokens, win_start) = match streaming_mask {
        Some((sink, window)) => (sink, query_pos.saturating_add(1).saturating_sub(window)),
        None => (0, 0),
    };

    for k_pos in 0..=query_pos {
        if streaming_mask.is_some() && k_pos >= sink_tokens && k_pos < win_start {
            continue;
        }
        let off = k_pos * head_dim;
        let k = &keys[off..off + head_dim];
        let v = &values[off..off + head_dim];

        let mut dot = 0.0_f32;
        for i in 0..head_dim {
            dot += query[i] * k[i];
        }
        let dist = (query_pos - k_pos) as f32;
        let mut score = dot * inv_sqrt_d - pos_decay * dist;
        if k_pos < 4 && pos_decay > 0.0 {
            score += 3.5;
        }
        if !score.is_finite() {
            score = -1e4;
        }

        if score > running_max {
            let rescale = if running_max.is_finite() {
                (running_max - score).exp()
            } else {
                0.0
            };
            for val in out.iter_mut() {
                *val *= rescale;
            }
            running_sum = running_sum * rescale + 1.0;
            for i in 0..head_dim {
                out[i] += v[i];
            }
            running_max = score;
        } else {
            let weight = (score - running_max).exp();
            running_sum += weight;
            for i in 0..head_dim {
                out[i] += weight * v[i];
            }
        }
    }

    if running_sum > 0.0 {
        let inv = 1.0 / running_sum;
        for val in out.iter_mut() {
            *val *= inv;
        }
    }
}

/// One GQA attention layer with deterministic linear projections (`W_q, W_k, W_v, W_o`)
/// and per-KV-head specialization (retrieval vs streaming behavior baked into weights).
#[derive(Clone, Debug)]
pub struct DualAttentionLayer {
    /// Model hidden dimension (`num_q_heads * head_dim`).
    pub d_model: usize,
    /// Number of query heads.
    pub num_q_heads: usize,
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Query projection `[num_q_heads * head_dim, d_model]`.
    w_q: Vec<f32>,
    /// Key projection `[num_kv_heads * head_dim, d_model]`.
    w_k: Vec<f32>,
    /// Value projection `[num_kv_heads * head_dim, d_model]`.
    w_v: Vec<f32>,
    /// Output projection `[d_model, num_q_heads * head_dim]`.
    w_o: Vec<f32>,
    /// Per-KV-head recency slope (`0.0` for retrieval heads, positive for streaming heads).
    kv_pos_decay: Vec<f32>,
    /// Underlying true behavioral specialization baked into this checkpoint's weights
    /// (used ONLY for post-hoc calibration recall diagnostics in tests, NEVER read by calibration).
    latent_true_roles: Vec<HeadRole>,
}

impl DualAttentionLayer {
    fn new_deterministic(
        d_model: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        true_roles: &[HeadRole],
        retrieval_slot_ids: &[usize],
        rng: &mut SplitMix64,
    ) -> Self {
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let group = num_q_heads / num_kv_heads;
        let key_dim = (d_model / 4).max(head_dim * 2);
        let key_half = (key_dim / 2).max(head_dim);
        let val_start = key_dim;
        let val_dim = (d_model / 2).max(head_dim);
        let loc_start = val_start + val_dim;
        let loc_dim = d_model.saturating_sub(loc_start).max(1);

        let mut w_q = vec![0.0_f32; q_dim * d_model];
        let mut w_k = vec![0.0_f32; kv_dim * d_model];
        let mut w_v = vec![0.0_f32; kv_dim * d_model];
        let mut w_o = vec![0.0_f32; d_model * q_dim];
        let mut kv_pos_decay = vec![0.0_f32; num_kv_heads];

        let scale_noise = 0.03_f32 / (d_model as f32).sqrt();
        // Asymmetric Q/K associative recall:
        // - W_k reads definition keys from `key_def [0..key_half)`
        // - W_q reads query cues from `query_cue [key_half..key_dim)`
        // - W_v and W_o read/write each retrieval head's dedicated slice of `val [val_start..loc_start)`
        // Because definition tokens carry zero query_cue, distractor tokens never copy distant
        // needle values across layers, and query tokens cannot cheat via self-attention.
        let key_slice_gain = ((key_half as f32) / (head_dim.max(1) as f32)).sqrt();
        let ret_qk_gain = 3.2 * key_slice_gain;

        for h in 0..num_kv_heads {
            let is_retrieval = true_roles[h] == HeadRole::Retrieval;
            // Streaming heads decay attention smoothly outside local window + sinks.
            kv_pos_decay[h] = if is_retrieval { 0.0 } else { 0.26 };
            let key_slot = (retrieval_slot_ids[h] * head_dim) % key_half.max(1);
            let val_base = val_start + ((retrieval_slot_ids[h] * head_dim) % val_dim.max(1));

            // Key & Value projections for KV head `h`.
            for r in 0..head_dim {
                let row_k = (h * head_dim + r) * d_model;
                let row_v = (h * head_dim + r) * d_model;
                for c in 0..d_model {
                    w_k[row_k + c] = rng.next_normal_f32() * scale_noise;
                    w_v[row_v + c] = rng.next_normal_f32() * scale_noise;
                }
                if is_retrieval {
                    let key_col = (key_slot + r) % key_half.max(1);
                    let val_col = val_start + ((val_base - val_start + r) % val_dim.max(1));
                    w_k[row_k + key_col] += ret_qk_gain;
                    w_v[row_v + val_col] += 2.4;
                } else {
                    // Streaming KV heads attend locally in the syntactic/sink subspace.
                    let loc_col = loc_start + ((h * 5 + r) % loc_dim);
                    w_k[row_k + loc_col] += 1.6;
                    w_v[row_v + loc_col] += 0.85;
                }
            }

            // Query & Output projections for the GQA query group mapped to KV head `h`.
            for g in 0..group {
                let qh = h * group + g;
                for r in 0..head_dim {
                    let row_q = (qh * head_dim + r) * d_model;
                    for c in 0..d_model {
                        w_q[row_q + c] = rng.next_normal_f32() * scale_noise;
                    }
                    if is_retrieval {
                        let query_col = key_half + ((key_slot + r) % key_half.max(1));
                        w_q[row_q + query_col] += ret_qk_gain;
                    } else {
                        let loc_col = loc_start + ((h * 5 + r) % loc_dim);
                        w_q[row_q + loc_col] += 1.6;
                    }
                }

                // Output projection W_O writes recalled values back into the value subspace
                // (or local features into the local subspace).
                for out_r in 0..d_model {
                    for r in 0..head_dim {
                        let col = qh * head_dim + r;
                        w_o[out_r * q_dim + col] = rng.next_normal_f32() * scale_noise;
                    }
                }
                for r in 0..head_dim {
                    let col = qh * head_dim + r;
                    if is_retrieval {
                        let val_row = val_start + ((val_base - val_start + r) % val_dim.max(1));
                        w_o[val_row * q_dim + col] += 1.45 / (group as f32);
                    } else {
                        let loc_row = loc_start + ((h * 5 + r) % loc_dim);
                        w_o[loc_row * q_dim + col] += 0.65 / (group as f32);
                    }
                }
            }
        }

        Self {
            d_model,
            num_q_heads,
            num_kv_heads,
            head_dim,
            w_q,
            w_k,
            w_v,
            w_o,
            kv_pos_decay,
            latent_true_roles: true_roles.to_vec(),
        }
    }

    /// Projects hidden vector `x` (`len == d_model`) into `q`, `k`, `v`.
    pub fn project_qkv(&self, x: &[f32], q: &mut [f32], k: &mut [f32], v: &mut [f32]) {
        matvec(
            &self.w_q,
            x,
            self.num_q_heads * self.head_dim,
            self.d_model,
            q,
        );
        matvec(
            &self.w_k,
            x,
            self.num_kv_heads * self.head_dim,
            self.d_model,
            k,
        );
        matvec(
            &self.w_v,
            x,
            self.num_kv_heads * self.head_dim,
            self.d_model,
            v,
        );
    }

    /// Projects a single KV-head's GQA output group (`group * head_dim`) through its slice of `W_O`.
    pub fn project_kv_group_output(
        &self,
        kv_head: usize,
        group_out: &[f32],
        out_d_model: &mut [f32],
    ) {
        let group = self.num_q_heads / self.num_kv_heads;
        let q_dim = self.num_q_heads * self.head_dim;
        let col_start = kv_head * group * self.head_dim;
        let width = group * self.head_dim;
        for (r, dst) in out_d_model.iter_mut().enumerate().take(self.d_model) {
            let row = &self.w_o[r * q_dim + col_start..r * q_dim + col_start + width];
            let mut acc = 0.0_f32;
            for i in 0..width {
                acc += row[i] * group_out[i];
            }
            *dst = acc;
        }
    }

    /// Projects concatenated attention head outputs `[num_q_heads * head_dim]` through `W_O`.
    pub fn project_output(&self, attn_concat: &[f32], out: &mut [f32]) {
        matvec(
            &self.w_o,
            attn_concat,
            self.d_model,
            self.num_q_heads * self.head_dim,
            out,
        );
    }

    /// Underlying ground-truth behavioral role of KV head `h` (diagnostics/tests only).
    #[must_use]
    pub fn diagnostic_true_role(&self, kv_head: usize) -> HeadRole {
        self.latent_true_roles[kv_head]
    }
}

/// Multi-layer causal GQA attention model used for offline calibration, real dual-cache
/// prefill/decode execution, and end-to-end long-context retrieval evaluation.
#[derive(Clone, Debug)]
pub struct DualAttentionModel {
    identity: ModelIdentity,
    d_model: usize,
    layers: Vec<DualAttentionLayer>,
    /// True fraction of retrieval heads baked into the model weights.
    true_retrieval_ratio: f32,
}

impl DualAttentionModel {
    /// Constructs a deterministic multi-layer GQA model whose head specialization is
    /// uniquely determined by `identity.model_checkpoint_id`.
    ///
    /// Two models with distinct `model_checkpoint_id` values have genuinely distinct
    /// retrieval-head positions, reflecting the real-world fact that DuoAttention head
    /// masks are checkpoint-specific and cannot be safely transferred across models.
    pub fn from_identity(
        identity: ModelIdentity,
        true_retrieval_ratio: f32,
    ) -> Result<Self, DuoAttnError> {
        identity.validate()?;
        if !(0.0..=1.0).contains(&true_retrieval_ratio) || true_retrieval_ratio.is_nan() {
            return Err(DuoAttnError::InvalidConfig(format!(
                "true_retrieval_ratio must be in [0.0, 1.0], got {true_retrieval_ratio}"
            )));
        }

        // Derive a deterministic checkpoint seed from the model_checkpoint_id via SHA-256.
        let digest = crate::sha256::Sha256::new();
        let mut h = digest;
        h.update_len_prefixed(b"hanzo-duoattn/model-weights/v1");
        h.update_len_prefixed(identity.model_checkpoint_id.as_bytes());
        h.update_u64(identity.num_attn_layers as u64);
        h.update_u64(identity.num_kv_heads as u64);
        h.update_u64(identity.head_dim as u64);
        let bytes = h.finalize();
        let seed = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let mut rng = SplitMix64::new(seed);

        let total_kv = identity.total_kv_heads();
        let target_ret = ((true_retrieval_ratio as f64) * (total_kv as f64))
            .round()
            .clamp(1.0, total_kv as f64) as usize;

        // Deterministic checkpoint-specific permutation of which heads are retrieval heads.
        let mut perm: Vec<usize> = (0..total_kv).collect();
        for i in (1..total_kv).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            perm.swap(i, j);
        }
        let mut roles = vec![HeadRole::Streaming; total_kv];
        let mut slot_ids = vec![0usize; total_kv];
        for (rank, &idx) in perm.iter().take(target_ret).enumerate() {
            roles[idx] = HeadRole::Retrieval;
            slot_ids[idx] = rank;
        }

        let d_model = identity.num_q_heads * identity.head_dim;
        let mut layers = Vec::with_capacity(identity.num_attn_layers);
        for l in 0..identity.num_attn_layers {
            let start = l * identity.num_kv_heads;
            let end = start + identity.num_kv_heads;
            layers.push(DualAttentionLayer::new_deterministic(
                d_model,
                identity.num_q_heads,
                identity.num_kv_heads,
                identity.head_dim,
                &roles[start..end],
                &slot_ids[start..end],
                &mut rng,
            ));
        }

        Ok(Self {
            identity,
            d_model,
            layers,
            true_retrieval_ratio,
        })
    }

    /// Returns the model's immutable identity.
    #[must_use]
    pub fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    /// Hidden dimension (`num_q_heads * head_dim`).
    #[must_use]
    pub const fn d_model(&self) -> usize {
        self.d_model
    }

    /// Fraction of heads that genuinely exhibit retrieval behavior in this checkpoint.
    #[must_use]
    pub const fn true_retrieval_ratio(&self) -> f32 {
        self.true_retrieval_ratio
    }

    /// Attention layers slice.
    #[must_use]
    pub fn layers(&self) -> &[DualAttentionLayer] {
        &self.layers
    }

    /// Ground-truth retrieval head coordinates for post-hoc calibration precision/recall auditing.
    #[must_use]
    pub fn diagnostic_true_roles(&self) -> Vec<HeadRole> {
        let mut out = Vec::with_capacity(self.identity.total_kv_heads());
        for layer in &self.layers {
            for h in 0..layer.num_kv_heads {
                out.push(layer.diagnostic_true_role(h));
            }
        }
        out
    }

    /// Runs a full causal forward pass over `token_embeddings` (`[seq_len, d_model]`),
    /// populating `cache` incrementally and returning the final token's hidden state `[d_model]`.
    pub fn forward_sequence(
        &self,
        token_embeddings: &[Vec<f32>],
        cache: &mut DualKvCache,
    ) -> Result<Vec<f32>, DuoAttnError> {
        if token_embeddings.is_empty() {
            return Err(DuoAttnError::InvalidConfig(
                "token_embeddings must be non-empty".to_string(),
            ));
        }
        let mut hidden = Vec::new();
        for (pos, emb) in token_embeddings.iter().enumerate() {
            hidden = self.forward_step(pos, emb, cache)?;
        }
        Ok(hidden)
    }

    /// Runs one autoregressive step at logical `position`, appending KV to `cache` and
    /// executing real multi-head GQA attention across every attention layer.
    pub fn forward_step(
        &self,
        position: usize,
        input_embedding: &[f32],
        cache: &mut DualKvCache,
    ) -> Result<Vec<f32>, DuoAttnError> {
        if input_embedding.len() != self.d_model {
            return Err(DuoAttnError::DimensionMismatch {
                context: "DualAttentionModel::forward_step(input_embedding)",
                expected: self.d_model,
                actual: input_embedding.len(),
            });
        }
        let mut x = input_embedding.to_vec();
        let q_dim = self.identity.num_q_heads * self.identity.head_dim;
        let kv_dim = self.identity.num_kv_heads * self.identity.head_dim;
        let d = self.identity.head_dim;
        let group = self.identity.gqa_group_size();

        let mut q = vec![0.0_f32; q_dim];
        let mut k = vec![0.0_f32; kv_dim];
        let mut v = vec![0.0_f32; kv_dim];
        let mut attn_concat = vec![0.0_f32; q_dim];
        let mut proj_out = vec![0.0_f32; self.d_model];

        for (l, layer) in self.layers.iter().enumerate() {
            let normed = rms_norm(&x);
            layer.project_qkv(&normed, &mut q, &mut k, &mut v);
            cache.append_layer_step(l, position, &k, &v)?;

            for qh in 0..self.identity.num_q_heads {
                let kvh = qh / group;
                let storage = cache.head_storage(l, kvh)?;
                let q_slice = &q[qh * d..(qh + 1) * d];
                let out_slice = &mut attn_concat[qh * d..(qh + 1) * d];
                sdpa_head_against_storage(
                    q_slice,
                    storage,
                    position,
                    layer.kv_pos_decay[kvh],
                    out_slice,
                )?;
            }

            layer.project_output(&attn_concat, &mut proj_out);
            for i in 0..self.d_model {
                x[i] += 0.35 * proj_out[i];
            }
        }
        Ok(x)
    }

    /// Computes per-layer, per-KV-head `(O_full - O_stream)` discrepancy vectors projected through `W_O`
    /// across evaluation positions `eval_positions` of a calibration sequence.
    ///
    /// Used by [`crate::calibrate::calibrate_model`] to construct the exact DuoAttention
    /// Gram matrices and optimize `alpha_{l,h} in [0, 1]` without inspecting `latent_true_roles`.
    pub fn collect_calibration_discrepancies(
        &self,
        token_embeddings: &[Vec<f32>],
        eval_positions: &[usize],
        sink_tokens: usize,
        window_tokens: usize,
    ) -> Result<Vec<Vec<Vec<Vec<f32>>>>, DuoAttnError> {
        let seq_len = token_embeddings.len();
        if seq_len == 0 {
            return Err(DuoAttnError::InvalidConfig(
                "calibration sequence must be non-empty".to_string(),
            ));
        }
        for emb in token_embeddings {
            if emb.len() != self.d_model {
                return Err(DuoAttnError::DimensionMismatch {
                    context: "collect_calibration_discrepancies(emb)",
                    expected: self.d_model,
                    actual: emb.len(),
                });
            }
        }

        let num_l = self.identity.num_attn_layers;
        let num_q = self.identity.num_q_heads;
        let num_kv = self.identity.num_kv_heads;
        let d = self.identity.head_dim;
        let group = self.identity.gqa_group_size();

        // Result shape: [layer][kv_head][eval_pos_idx][d_model]
        let mut discrepancies = vec![vec![Vec::with_capacity(eval_positions.len()); num_kv]; num_l];

        // Propagate hidden states through the exact full-attention teacher trajectory
        // (matching DuoAttention teacher-forced distillation, arXiv:2410.10819 Eq. 3).
        let mut hidden_seq: Vec<Vec<f32>> = token_embeddings.to_vec();

        for (l, layer) in self.layers.iter().enumerate() {
            // Project Q, K, V for every token in the sequence.
            let mut q_all = vec![vec![0.0_f32; seq_len * d]; num_q];
            let mut k_all = vec![vec![0.0_f32; seq_len * d]; num_kv];
            let mut v_all = vec![vec![0.0_f32; seq_len * d]; num_kv];

            let mut q_buf = vec![0.0_f32; num_q * d];
            let mut k_buf = vec![0.0_f32; num_kv * d];
            let mut v_buf = vec![0.0_f32; num_kv * d];

            for (pos, h_vec) in hidden_seq.iter().enumerate() {
                let normed = rms_norm(h_vec);
                layer.project_qkv(&normed, &mut q_buf, &mut k_buf, &mut v_buf);
                for qh in 0..num_q {
                    q_all[qh][pos * d..(pos + 1) * d].copy_from_slice(&q_buf[qh * d..(qh + 1) * d]);
                }
                for kvh in 0..num_kv {
                    k_all[kvh][pos * d..(pos + 1) * d]
                        .copy_from_slice(&k_buf[kvh * d..(kvh + 1) * d]);
                    v_all[kvh][pos * d..(pos + 1) * d]
                        .copy_from_slice(&v_buf[kvh * d..(kvh + 1) * d]);
                }
            }

            // Record full-vs-streaming discrepancy for each KV head at eval_positions.
            let mut full_group = vec![0.0_f32; group * d];
            let mut stream_group = vec![0.0_f32; group * d];
            let mut diff_group = vec![0.0_f32; group * d];
            let mut diff_proj = vec![0.0_f32; self.d_model];

            for &pos in eval_positions {
                if pos >= seq_len {
                    continue;
                }
                for kvh in 0..num_kv {
                    let decay = layer.kv_pos_decay[kvh];
                    for g in 0..group {
                        let qh = kvh * group + g;
                        let q_slice = &q_all[qh][pos * d..(pos + 1) * d];
                        sdpa_head_masked(
                            q_slice,
                            &k_all[kvh],
                            &v_all[kvh],
                            d,
                            pos,
                            decay,
                            None,
                            &mut full_group[g * d..(g + 1) * d],
                        );
                        sdpa_head_masked(
                            q_slice,
                            &k_all[kvh],
                            &v_all[kvh],
                            d,
                            pos,
                            decay,
                            Some((sink_tokens, window_tokens)),
                            &mut stream_group[g * d..(g + 1) * d],
                        );
                    }
                    for i in 0..group * d {
                        diff_group[i] = full_group[i] - stream_group[i];
                    }
                    layer.project_kv_group_output(kvh, &diff_group, &mut diff_proj);
                    discrepancies[l][kvh].push(diff_proj.clone());
                }
            }

            // Advance hidden_seq using teacher (full) attention for next layer.
            if l + 1 < num_l {
                let mut attn_concat = vec![0.0_f32; num_q * d];
                let mut proj_out = vec![0.0_f32; self.d_model];
                for (pos, h_vec) in hidden_seq.iter_mut().enumerate() {
                    for qh in 0..num_q {
                        let kvh = qh / group;
                        let q_slice = &q_all[qh][pos * d..(pos + 1) * d];
                        sdpa_head_masked(
                            q_slice,
                            &k_all[kvh],
                            &v_all[kvh],
                            d,
                            pos,
                            layer.kv_pos_decay[kvh],
                            None,
                            &mut attn_concat[qh * d..(qh + 1) * d],
                        );
                    }
                    layer.project_output(&attn_concat, &mut proj_out);
                    for i in 0..self.d_model {
                        h_vec[i] += 0.35 * proj_out[i];
                    }
                }
            }
        }

        Ok(discrepancies)
    }
}

/// RMSNorm helper (`eps = 1e-6`).
fn rms_norm(x: &[f32]) -> Vec<f32> {
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / (x.len().max(1) as f32);
    let inv_rms = 1.0_f32 / (mean_sq + 1e-6).sqrt();
    x.iter().map(|&v| v * inv_rms).collect()
}

/// Matrix-vector product `out = mat * x` where `mat` is row-major `[rows, cols]`.
fn matvec(mat: &[f32], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for (r, dst) in out.iter_mut().enumerate().take(rows) {
        let row = &mat[r * cols..(r + 1) * cols];
        let mut acc = 0.0_f32;
        for c in 0..cols {
            acc += row[c] * x[c];
        }
        *dst = acc;
    }
}
