//! Warm 20-turn coding-agent benchmark on Zen6 (GGUF + CUDA + PagedAttention).
//!
//! Compares five end-to-end execution modes sequentially (one loaded engine at a time):
//! 1. `plain_nocache`: KV prefix cache off, no speculation
//! 2. `cached_plain`: KV prefix cache on (`n = 16`), no speculation
//! 3. `cached_mtp_unguarded`: KV prefix cache on + unguarded self-MTP head (`n_predict = 2`)
//! 4. `cached_prompt_lookup`: KV prefix cache on + standalone n-gram prompt lookup (`1..=8, gamma=6`)
//! 5. `cached_adaptive_full`: KV prefix cache on + guarded self-MTP (`GuardedDraft`) + suffix lookup
//!    (`2..=8, gamma=6`) + `SA_SPEC` bypass (`threshold = 4`)

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use either::Either;
use hanzo_engine::speculative::{stats as spec_stats, Lookup};
use hanzo_engine::{
    Hanzo, ModelDType, ModelSelected, MtpConfig, NormalRequest, Request, RequestMessage, Response,
    SamplingParams, TokenSource, Usage,
};
use hanzo_server_core::server::ServerBuilder;
use indexmap::IndexMap;
use serde::Serialize;
use tokio::sync::mpsc::channel;

const MODEL_DIR: &str = "/data/models/zen6-source/4B-gguf";
const MODEL_FILE: &str = "Ternary-Bonsai-2-4B-PQ2_0-MTP-Q8_0.gguf";
const DEFAULT_OUT_JSON: &str = "zen6_20turn_results.json";
const MAX_SEQ_LEN: usize = 8192;
const MAX_GEN_TOKENS: usize = 96;

#[derive(Clone)]
struct BenchConfigSpec {
    name: &'static str,
    prefix_cache_n: usize,
    mtp: bool,
    adaptive_guard: bool,
    lookup_max_ngram: Option<usize>,
    lookup_gamma: usize,
    sa_bypass_threshold: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct TurnMetrics {
    turn: usize,
    ttft_ms: f64,
    wall_ms: f64,
    engine_gpu_ms: f64,
    prompt_tokens: usize,
    cached_prompt_tokens: usize,
    completion_tokens: usize,
    usage_draft_tokens: usize,
    usage_draft_accepted: usize,
    itl_samples_ms: Vec<f64>,
    output_text: String,
}

#[derive(Debug, Clone, Serialize)]
struct ConfigSummary {
    name: String,
    prefix_cache_n: usize,
    mtp: bool,
    adaptive_guard: bool,
    lookup_max_ngram: Option<usize>,
    lookup_gamma: usize,
    sa_bypass_threshold: Option<usize>,
    ttft_p50_ms: f64,
    ttft_p95_ms: f64,
    ttft_mean_ms: f64,
    itl_p50_ms: f64,
    itl_p95_ms: f64,
    itl_mean_ms: f64,
    prose_itl_p50_ms: f64,
    edit_itl_p50_ms: f64,
    total_wall_ms: f64,
    engine_gpu_ms: f64,
    total_prompt_tokens: usize,
    total_cached_prompt_tokens: usize,
    total_completion_tokens: usize,
    usage_draft_tokens: usize,
    usage_draft_accepted: usize,
    verify_rounds: u64,
    proposed_sum: u64,
    accepted_sum: u64,
    sa_bypass_rounds: u64,
    guard_skips: u64,
    exact_turns_vs_cached_plain: usize,
    turn_first_diff_chars: Vec<Option<usize>>,
    turns: Vec<TurnMetrics>,
}

#[derive(Debug, Serialize)]
struct BenchmarkOutput {
    model: String,
    turns_per_config: usize,
    prose_turns: usize,
    edit_turns: usize,
    max_gen_tokens: usize,
    note: String,
    exact_turns_vs_cached_plain: HashMap<String, usize>,
    configs: Vec<ConfigSummary>,
}

fn chat_msg(role: &str, content: &str) -> IndexMap<String, hanzo_engine::MessageContent> {
    let mut m = IndexMap::new();
    m.insert("role".to_string(), Either::Left(role.to_string()));
    m.insert("content".to_string(), Either::Left(content.to_string()));
    m
}

fn percentile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn developer_turn_prompts() -> Vec<String> {
    vec![
        // Turns 1..=10: Diagnostic / architecture / telemetry Q&A (ungrounded prose -> tests GuardedDraft cooldown + CUDA decode graph recovery)
        "Explain in three concise sentences why prefix tokenization caching in `PromptTokenCache` automatically enters bypass mode on cold misses when cache lookup overhead exceeds direct BPE cost.".into(),
        "Why does setting `RAYON_NUM_THREADS=1` without switching away from multi-threaded PCRE2 fail on inputs longer than 16 KiB, and how did `hanzo-tokenizer` eliminate that failure mode?".into(),
        "Explain the algorithmic difference between textbook earliest-position suffix automatons and `SuffixAutomaton`'s `last_end` / `last_2gram` design for coding-agent completions.".into(),
        "Why does checking `draft_needs_target_hiddens()` instead of `self.draft_proposer.is_some()` restore CUDA decode graph performance during `GuardedDraft` cooldown or standalone `PromptLookup`?".into(),
        "Give a brief engineering interpretation of why `UpsampleNearest2D` on NEON CPU delivered ~1.0x speedup (`2.20 ms` vs `2.19 ms`) compared to `1080p`/`4K` RGB-to-CHW vision preprocessing (`3.5x-3.9x`).".into(),
        "Why do FP16 and BF16 GPU reductions occasionally produce greedy token tie-breaks between plain (`M=1`) prefill/decode and multi-token (`M=3`) speculative verification without indicating a verifier bug?".into(),
        "Explain why `SA_SPEC` must still send every suffix-proposed token through target-model logits verification instead of unconditionally committing repeated spans from context.".into(),
        "What does `MtpConfig::adaptive_guard = true` achieve over unguarded MTP on open-ended explanatory developer turns where draft acceptance averages below `1.2` tokens/round?".into(),
        "Why does `Extended::observe_outcomes` update `GuardedDraft`'s verification baseline before `try_sa_bypass` runs instead of letting `SA_SPEC` rounds arm `GuardedDraft`?".into(),
        "Summarize in three short points how PagedAttention KV prefix caching (`prefix_cache_n = 16`) affects TTFT versus decode-graph recovery affecting steady-state ITL in a 20-turn coding session.".into(),
        // Turns 11..=20: Grounded code-editing / verbatim refactor tasks (repeated context spans -> exercises PromptLookup + SA_SPEC draft bypass)
        "Edit ONLY line 7 (`min_accepted` -> `1.5`) in this Rust function. Output ONLY the complete updated Rust code block verbatim with no extra commentary:\n```rust\npub fn default_mtp_guard() -> AdaptiveSpeculativeGuard {\n    AdaptiveSpeculativeGuard::new(\n        1.2,\n        2,\n        16,\n    )\n    .with_cross_seq_carry(true)\n}\n```".into(),
        "Update `resolve_mtp_config` below ONLY by adding `.with_adaptive_guard(mtp_config.adaptive_guard)` right after `.with_sa_bypass_threshold(...)`. Return ONLY the complete Rust code block verbatim:\n```rust\nfn resolve_mtp_config(\n    mut mtp_config: MtpConfig,\n    prompt_lookup_ngram: Option<usize>,\n    gamma: usize,\n    sa_bypass_threshold: Option<usize>,\n) -> anyhow::Result<MtpConfig> {\n    let lookup = Self::mtp_lookup(prompt_lookup_ngram, gamma)?.or(mtp_config.lookup);\n    mtp_config = mtp_config.with_lookup(lookup);\n    if sa_bypass_threshold.is_some() {\n        mtp_config = mtp_config.with_sa_bypass_threshold(sa_bypass_threshold);\n    }\n    Ok(mtp_config)\n}\n```".into(),
        "Update `sync_model_spec_capture` below ONLY by replacing `let capture = self.draft_proposer.is_some();` with `let capture = self.draft_needs_target_hiddens();`. Output ONLY the complete Rust code block verbatim:\n```rust\nfn sync_model_spec_capture(&self) {\n    let capture = self.draft_proposer.is_some();\n    match self.model {\n        Model::Qwen35(ref model) => model.set_store_spec(capture),\n        Model::Zen6(ref model) => model.set_store_spec(capture),\n        Model::Deepseek2(ref model) => model.set_store_spec_hidden(capture),\n        Model::Deepseek4(ref model) => model.set_store_spec_hidden(capture),\n        _ => {}\n    }\n}\n```".into(),
        "Refactor `draft_counts` below ONLY by renaming the local binding `group` to `seq_group`. Output ONLY the complete updated Rust code block verbatim:\n```rust\n/// Cumulative `(total_draft_toks, total_draft_accepted)` counters for this sequence group.\npub(crate) fn draft_counts(&self) -> (usize, usize) {\n    let group = get_mut_group!(self);\n    (\n        group.total_draft_toks,\n        group.total_draft_accepted,\n    )\n}\n```".into(),
        "Update `SuffixAutomaton::sync_prefix` below ONLY by changing `.max(16)` to `.max(32)`. Output ONLY the complete updated Rust code block verbatim:\n```rust\npub fn sync_prefix(&mut self, tokens: &[u32]) {\n    let reserve = tokens.len().saturating_mul(2).max(16);\n    if self.states.capacity() < reserve {\n        self.states.reserve(reserve - self.states.capacity());\n    }\n    let lcp = common_prefix_len(&self.prefix, tokens);\n    if lcp < self.prefix.len() {\n        self.truncate(lcp);\n    }\n    self.extend_slice(&tokens[lcp..]);\n}\n```".into(),
        "Update `PromptTokenCache` telemetry below ONLY by changing `\"bypassed\": false` to `\"bypassed\": true`. Output ONLY the complete JSON code block verbatim:\n```json\n{\n  \"model\": \"DeepSeek-V3.2\",\n  \"turns\": 20,\n  \"prefix_hits\": 0,\n  \"bypassed\": false,\n  \"cold_ns_per_kib\": 22500.0,\n  \"warm_ns_per_kib\": 87900.0\n}\n```".into(),
        "Update `SpeculativePipelineExt` below ONLY by adding `fn needs_hiddens_now(&self) -> bool;` right after `speculative_proposal_len`. Output ONLY the complete Rust trait snippet verbatim:\n```rust\npub trait SpeculativePipelineExt: Pipeline {\n    fn has_speculative_proposer(&self) -> bool;\n    fn speculative_proposal_len(&self) -> Option<usize>;\n    fn speculative_target_hiddens(\n        &self,\n        rows: &[(usize, usize)],\n    ) -> Result<Option<Tensor>>;\n}\n```".into(),
        "Update `check_lsa_gate` below ONLY by changing `.saturating_add(1)` to `.saturating_add(4)`. Output ONLY the complete Rust snippet verbatim:\n```rust\nlet lsa_active = crate::attention::LsaConfig::global().is_some_and(|lsa_cfg| {\n    let max_kv_len = metadata.max_context_len.unwrap_or_else(|| {\n        seqlen_offsets\n            .iter()\n            .copied()\n            .max()\n            .unwrap_or(0)\n            .saturating_add(1)\n    });\n    lsa_cfg.active_for_seq_len(max_kv_len)\n});\n```".into(),
        "Update `MtpConfig::new` below ONLY by changing `sa_bypass_threshold: None` to `sa_bypass_threshold: Some(4)`. Output ONLY the complete Rust snippet verbatim:\n```rust\nimpl MtpConfig {\n    pub fn new(model: impl Into<String>, n_predict: Option<usize>) -> Self {\n        Self {\n            model: model.into(),\n            n_predict,\n            lookup: None,\n            sa_bypass_threshold: None,\n            adaptive_guard: true,\n        }\n    }\n}\n```".into(),
        "Update the `BenchConfigSpec` below ONLY by changing `lookup_gamma: 4` to `lookup_gamma: 6`. Output ONLY the complete Rust struct literal verbatim:\n```rust\nBenchConfigSpec {\n    name: \"cached_adaptive_full\",\n    prefix_cache_n: 16,\n    mtp: true,\n    adaptive_guard: true,\n    lookup_max_ngram: Some(8),\n    lookup_gamma: 4,\n    sa_bypass_threshold: Some(4),\n}\n```".into(),
    ]
}

async fn build_engine(spec: &BenchConfigSpec) -> Result<Arc<Hanzo>> {
    let mut builder = ServerBuilder::new()
        .with_model(ModelSelected::GGUF {
            tok_model_id: None,
            quantized_model_id: MODEL_DIR.to_string(),
            quantized_filename: MODEL_FILE.to_string(),
            dtype: ModelDType::Auto,
            topology: None,
            rope_yarn_factor: None,
            max_seq_len: MAX_SEQ_LEN,
            max_batch_size: 1,
        })
        .with_max_seqs(1)
        .with_no_kv_cache(false)
        .with_token_source(TokenSource::None)
        .with_interactive_mode(false)
        .with_prefix_cache_n(spec.prefix_cache_n)
        .set_paged_attn(Some(true))
        .with_paged_ctxt_len(MAX_SEQ_LEN);

    if spec.mtp {
        let mut cfg = MtpConfig::new("self", Some(2)).with_adaptive_guard(spec.adaptive_guard);
        if let Some(ngram_max) = spec.lookup_max_ngram {
            let lookup = Lookup::new(2, ngram_max, spec.lookup_gamma)?;
            cfg = cfg.with_lookup(Some(lookup));
        }
        if let Some(thresh) = spec.sa_bypass_threshold {
            cfg = cfg.with_sa_bypass_threshold(Some(thresh));
        }
        builder = builder.with_mtp_config(cfg);
    } else if spec.lookup_max_ngram.is_some() {
        builder = builder.with_prompt_lookup_optional(spec.lookup_max_ngram, spec.lookup_gamma);
    }

    builder
        .build()
        .await
        .with_context(|| format!("building Zen6 engine for {}", spec.name))
}

async fn run_streaming_turn(
    hanzo: &Arc<Hanzo>,
    messages: Vec<IndexMap<String, hanzo_engine::MessageContent>>,
    max_len: usize,
) -> Result<TurnMetricsWithoutIndex> {
    let mut sp = SamplingParams::deterministic();
    sp.max_len = Some(max_len);

    let sender = hanzo.get_sender(None).context("engine sender")?;
    let (tx, mut rx) = channel(128);
    let mut req = NormalRequest::new_simple(
        RequestMessage::Chat {
            messages,
            enable_thinking: Some(false),
            reasoning_effort: None,
            preserve_thinking: None,
        },
        sp,
        tx,
        hanzo.next_request_id(),
        None,
        None,
    );
    req.is_streaming = true;

    let t_send = Instant::now();
    sender
        .send(Request::Normal(Box::new(req)))
        .await
        .map_err(|e| anyhow::anyhow!("send request failed: {e}"))?;

    let mut output_text = String::new();
    let mut first_content_at: Option<Instant> = None;
    let mut prev_content_at: Option<Instant> = None;
    let mut itl_samples_ms = Vec::new();
    let mut final_usage: Option<Usage> = None;

    while let Some(resp) = rx.recv().await {
        match resp {
            Response::Chunk(chunk) => {
                let now = Instant::now();
                if let Some(u) = chunk.usage {
                    final_usage = Some(u);
                }
                let mut done = false;
                for choice in &chunk.choices {
                    if let Some(delta) = choice.delta.content.as_deref() {
                        if !delta.is_empty() {
                            if first_content_at.is_none() {
                                first_content_at = Some(now);
                            } else if let Some(prev) = prev_content_at {
                                itl_samples_ms
                                    .push(now.duration_since(prev).as_secs_f64() * 1000.0);
                            }
                            prev_content_at = Some(now);
                            output_text.push_str(delta);
                        }
                    }
                    if choice.finish_reason.is_some() {
                        done = true;
                    }
                }
                if done {
                    break;
                }
            }
            Response::Done(done) => {
                let now = Instant::now();
                first_content_at.get_or_insert(now);
                if let Some(text) = done
                    .choices
                    .first()
                    .and_then(|c| c.message.content.as_deref())
                {
                    output_text = text.to_string();
                }
                final_usage = Some(done.usage);
                break;
            }
            Response::ValidationError(e) | Response::InternalError(e) => {
                bail!("engine returned error: {e}");
            }
            Response::ModelError(msg, _) => {
                bail!("model error: {msg}");
            }
            _ => {}
        }
    }

    let t_done = Instant::now();
    let wall_ms = t_done.duration_since(t_send).as_secs_f64() * 1000.0;
    let ttft_ms = first_content_at
        .unwrap_or(t_done)
        .duration_since(t_send)
        .as_secs_f64()
        * 1000.0;
    let usage = final_usage.context("missing terminal Usage on streaming response")?;
    let engine_gpu_ms =
        (usage.total_prompt_time_sec as f64 + usage.total_completion_time_sec as f64) * 1000.0;

    Ok(TurnMetricsWithoutIndex {
        ttft_ms,
        wall_ms,
        engine_gpu_ms,
        prompt_tokens: usage.prompt_tokens,
        cached_prompt_tokens: usage.cached_prompt_tokens,
        completion_tokens: usage.completion_tokens,
        usage_draft_tokens: usage.draft_tokens,
        usage_draft_accepted: usage.draft_accepted,
        itl_samples_ms,
        output_text,
    })
}

struct TurnMetricsWithoutIndex {
    ttft_ms: f64,
    wall_ms: f64,
    engine_gpu_ms: f64,
    prompt_tokens: usize,
    cached_prompt_tokens: usize,
    completion_tokens: usize,
    usage_draft_tokens: usize,
    usage_draft_accepted: usize,
    itl_samples_ms: Vec<f64>,
    output_text: String,
}

async fn run_config(spec: &BenchConfigSpec, prompts: &[String]) -> Result<ConfigSummary> {
    eprintln!("\n=== Building engine for `{}` ===", spec.name);
    let hanzo = build_engine(spec).await?;

    // Unrelated warmup request to warm CUDA kernels/allocators without seeding conversation KV prefix.
    let warmup_msgs = vec![
        chat_msg("system", "Respond concisely with one token."),
        chat_msg("user", "Warmup check: reply READY."),
    ];
    let _ = run_streaming_turn(&hanzo, warmup_msgs, 16).await?;

    let stats_before = spec_stats::snapshot();

    let mut messages = vec![chat_msg(
        "system",
        "You are a concise Rust and ML systems engineer. Give brief, technically accurate answers referencing the exact code and identifiers provided.",
    )];

    let mut turns = Vec::with_capacity(prompts.len());
    let mut all_ttft = Vec::with_capacity(prompts.len());
    let mut all_itl = Vec::new();
    let mut total_wall_ms = 0.0;
    let mut engine_gpu_ms = 0.0;
    let mut total_prompt_tokens = 0usize;
    let mut total_cached_prompt_tokens = 0usize;
    let mut total_completion_tokens = 0usize;
    let mut usage_draft_tokens = 0usize;
    let mut usage_draft_accepted = 0usize;

    for (turn_idx, user_prompt) in prompts.iter().enumerate() {
        messages.push(chat_msg("user", user_prompt));
        let res = run_streaming_turn(&hanzo, messages.clone(), MAX_GEN_TOKENS).await?;
        anyhow::ensure!(
            res.completion_tokens > 0,
            "config `{}` turn {} produced 0 completion tokens",
            spec.name,
            turn_idx
        );
        messages.push(chat_msg("assistant", &res.output_text));

        all_ttft.push(res.ttft_ms);
        all_itl.extend_from_slice(&res.itl_samples_ms);
        total_wall_ms += res.wall_ms;
        engine_gpu_ms += res.engine_gpu_ms;
        total_prompt_tokens += res.prompt_tokens;
        total_cached_prompt_tokens += res.cached_prompt_tokens;
        total_completion_tokens += res.completion_tokens;
        usage_draft_tokens += res.usage_draft_tokens;
        usage_draft_accepted += res.usage_draft_accepted;

        turns.push(TurnMetrics {
            turn: turn_idx + 1,
            ttft_ms: res.ttft_ms,
            wall_ms: res.wall_ms,
            engine_gpu_ms: res.engine_gpu_ms,
            prompt_tokens: res.prompt_tokens,
            cached_prompt_tokens: res.cached_prompt_tokens,
            completion_tokens: res.completion_tokens,
            usage_draft_tokens: res.usage_draft_tokens,
            usage_draft_accepted: res.usage_draft_accepted,
            itl_samples_ms: res.itl_samples_ms,
            output_text: res.output_text,
        });
    }

    let stats_after = spec_stats::snapshot();
    drop(hanzo);
    tokio::time::sleep(Duration::from_millis(500)).await;

    let split_prose = turns.len().min(10);
    let prose_itl: Vec<f64> = turns[..split_prose]
        .iter()
        .flat_map(|t| t.itl_samples_ms.iter().copied())
        .collect();
    let edit_itl: Vec<f64> = turns[split_prose..]
        .iter()
        .flat_map(|t| t.itl_samples_ms.iter().copied())
        .collect();

    Ok(ConfigSummary {
        name: spec.name.to_string(),
        prefix_cache_n: spec.prefix_cache_n,
        mtp: spec.mtp,
        adaptive_guard: spec.adaptive_guard,
        lookup_max_ngram: spec.lookup_max_ngram,
        lookup_gamma: spec.lookup_gamma,
        sa_bypass_threshold: spec.sa_bypass_threshold,
        ttft_p50_ms: percentile(&all_ttft, 0.50),
        ttft_p95_ms: percentile(&all_ttft, 0.95),
        ttft_mean_ms: mean(&all_ttft),
        itl_p50_ms: percentile(&all_itl, 0.50),
        itl_p95_ms: percentile(&all_itl, 0.95),
        itl_mean_ms: mean(&all_itl),
        prose_itl_p50_ms: percentile(&prose_itl, 0.50),
        edit_itl_p50_ms: percentile(&edit_itl, 0.50),
        total_wall_ms,
        engine_gpu_ms,
        total_prompt_tokens,
        total_cached_prompt_tokens,
        total_completion_tokens,
        usage_draft_tokens,
        usage_draft_accepted,
        verify_rounds: stats_after
            .verify_rounds
            .saturating_sub(stats_before.verify_rounds),
        proposed_sum: stats_after
            .proposed_sum
            .saturating_sub(stats_before.proposed_sum),
        accepted_sum: stats_after
            .accepted_sum
            .saturating_sub(stats_before.accepted_sum),
        sa_bypass_rounds: stats_after
            .sa_bypass_rounds
            .saturating_sub(stats_before.sa_bypass_rounds),
        guard_skips: stats_after
            .guard_skips
            .saturating_sub(stats_before.guard_skips),
        exact_turns_vs_cached_plain: 0,
        turn_first_diff_chars: Vec::new(),
        turns,
    })
}

fn first_diff_char(a: &str, b: &str) -> Option<usize> {
    if a == b {
        return None;
    }
    let idx = a
        .chars()
        .zip(b.chars())
        .take_while(|(ca, cb)| ca == cb)
        .count();
    Some(idx)
}

#[tokio::main]
async fn main() -> Result<()> {
    let model_path = PathBuf::from(MODEL_DIR).join(MODEL_FILE);
    anyhow::ensure!(
        model_path.is_file(),
        "missing Zen6 model GGUF at {}",
        model_path.display()
    );

    let prompts = developer_turn_prompts();
    anyhow::ensure!(prompts.len() == 20, "expected 20 developer turns");

    let specs = [
        BenchConfigSpec {
            name: "plain_nocache",
            prefix_cache_n: 0,
            mtp: false,
            adaptive_guard: false,
            lookup_max_ngram: None,
            lookup_gamma: 0,
            sa_bypass_threshold: None,
        },
        BenchConfigSpec {
            name: "cached_plain",
            prefix_cache_n: 16,
            mtp: false,
            adaptive_guard: false,
            lookup_max_ngram: None,
            lookup_gamma: 0,
            sa_bypass_threshold: None,
        },
        BenchConfigSpec {
            name: "cached_mtp_unguarded",
            prefix_cache_n: 16,
            mtp: true,
            adaptive_guard: false,
            lookup_max_ngram: None,
            lookup_gamma: 0,
            sa_bypass_threshold: None,
        },
        BenchConfigSpec {
            name: "cached_prompt_lookup",
            prefix_cache_n: 16,
            mtp: false,
            adaptive_guard: true,
            lookup_max_ngram: Some(8),
            lookup_gamma: 6,
            sa_bypass_threshold: None,
        },
        BenchConfigSpec {
            name: "cached_adaptive_full",
            prefix_cache_n: 16,
            mtp: true,
            adaptive_guard: true,
            lookup_max_ngram: Some(8),
            lookup_gamma: 6,
            sa_bypass_threshold: Some(4),
        },
    ];

    let mut summaries = Vec::with_capacity(specs.len());
    let mut outputs_by_cfg: HashMap<String, Vec<String>> = HashMap::new();

    for spec in &specs {
        let summary = run_config(spec, &prompts).await?;
        let texts: Vec<String> = summary
            .turns
            .iter()
            .map(|t| t.output_text.clone())
            .collect();
        outputs_by_cfg.insert(summary.name.clone(), texts);
        summaries.push(summary);
    }

    let cached_plain_out = outputs_by_cfg
        .get("cached_plain")
        .context("missing cached_plain outputs")?
        .clone();
    let mut exact_turns_vs_cached_plain = HashMap::new();
    for summary in &mut summaries {
        let diffs: Vec<Option<usize>> = summary
            .turns
            .iter()
            .zip(cached_plain_out.iter())
            .map(|(turn, plain_text)| first_diff_char(&turn.output_text, plain_text))
            .collect();
        let exact = diffs.iter().filter(|d| d.is_none()).count();
        summary.exact_turns_vs_cached_plain = exact;
        summary.turn_first_diff_chars = diffs;
        exact_turns_vs_cached_plain.insert(summary.name.clone(), exact);
    }

    println!("\n=== Warm Zen6 20-Turn Developer Agent Benchmark (CUDA) ===");
    println!(
        "{:<22} | {:>13} | {:>13} | {:>11} | {:>9} | {:>9} | {:>12} | {:>15} | {:>9} | {:>5}",
        "Config",
        "TTFT p50/p95",
        "ITL p50/p95",
        "Prose/Edit",
        "Wall ms",
        "GPU ms",
        "Prompt/Cache",
        "Draft Acc/Prop",
        "SA/Guard",
        "Match"
    );
    println!("{}", "-".repeat(142));
    for s in &summaries {
        println!(
            "{:<22} | {:>5.1}/{:>6.1} | {:>5.2}/{:>6.2} | {:>5.2}/{:>5.2} | {:>9.1} | {:>9.1} | {:>5}/{:>6} | {:>4}/{:>4} (r={:>3}) | {:>4}/{:>4} | {:>2}/20",
            s.name,
            s.ttft_p50_ms,
            s.ttft_p95_ms,
            s.itl_p50_ms,
            s.itl_p95_ms,
            s.prose_itl_p50_ms,
            s.edit_itl_p50_ms,
            s.total_wall_ms,
            s.engine_gpu_ms,
            s.total_prompt_tokens,
            s.total_cached_prompt_tokens,
            s.accepted_sum,
            s.proposed_sum,
            s.verify_rounds,
            s.sa_bypass_rounds,
            s.guard_skips,
            s.exact_turns_vs_cached_plain,
        );
    }
    println!(
        "Greedy turn matches vs cached_plain: {:?} (note: FP16/BF16 M=1 vs M=3 verification chunk-vs-serial reduction tie-breaks can diverge on long turns; see quantized_zen6_mtp.rs)",
        exact_turns_vs_cached_plain
    );

    let out_path = std::env::var("ZEN6_BENCH_OUT").unwrap_or_else(|_| DEFAULT_OUT_JSON.to_string());
    let report = BenchmarkOutput {
        model: model_path.display().to_string(),
        turns_per_config: prompts.len(),
        prose_turns: 10,
        edit_turns: 10,
        max_gen_tokens: MAX_GEN_TOKENS,
        note: "Turns 1-10 are diagnostic/explanatory prose; Turns 11-20 are grounded verbatim Rust/JSON edits. Under FP16/BF16 CUDA reductions, batched verify (M>1) and cached vs uncached prefill can break near-ties differently from M=1 decode without indicating verifier corruption.".to_string(),
        exact_turns_vs_cached_plain,
        configs: summaries,
    };
    std::fs::write(&out_path, serde_json::to_string_pretty(&report)?)
        .with_context(|| format!("writing benchmark JSON to {out_path}"))?;
    println!("Saved JSON report: {out_path}");

    Ok(())
}
