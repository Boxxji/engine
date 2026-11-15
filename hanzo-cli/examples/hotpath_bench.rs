// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: MIT

//! CPU release benchmark harness for Hanzo Engine hot paths:
//! 1. Clean-room `hanzo-tokenizer` + `PromptTokenCache` vs `tokenizers` (single-shot & 20-turn)
//! 2. `O(1)` incremental `SuffixAutomaton` + `SA_SPEC` bypass vs naive n-gram scan
//! 3. Cached vs uncached `minijinja` 12-turn + 2-tool chat template compilation/rendering
//! 4. `hanzo-vision` SIMD RGB resize + planar CHW normalization (`1080p` & `4K`)
//! 5. `hanzo-kv-fabric` epoch-fenced KV directory + `HKV1` TCP KV block transport

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use hanzo_kv_fabric::{
    compute_block_digest, kv_block_hash, EpochFencedKvDirectory, InMemoryKvBackend, KvBlockMeta,
    KvBlockTransport,
};
use hanzo_ml::Device;
use hanzo_tokenizer::{PromptTokenCache, Tokenizer as FastTokenizer};
use hanzo_vision::{
    simd_resize_rgb, ApplyTransforms, ImageTransform, Normalize, ToNormalizedTensor, ToTensor,
    Transforms,
};
use image::{imageops::FilterType, DynamicImage, RgbImage};
use minijinja::{context, Environment};
use serde::Serialize;
use tokenizers::Tokenizer as HfTokenizer;

fn percentile_us(samples: &mut [Duration], pct: f64) -> f64 {
    samples.sort_unstable();
    let idx = (((samples.len() - 1) as f64) * pct).round() as usize;
    samples[idx].as_secs_f64() * 1_000_000.0
}

fn percentile_ms(samples: &mut [Duration], pct: f64) -> f64 {
    percentile_us(samples, pct) / 1_000.0
}

fn naive_lookup(
    toks: &[u32],
    ngram_min: usize,
    ngram_max: usize,
    gamma: usize,
) -> (usize, Vec<u32>) {
    let len = toks.len();
    let longest = ngram_max.min(len.saturating_sub(1));
    for n in (ngram_min..=longest).rev() {
        let needle = &toks[len - n..];
        for start in (0..len - n).rev() {
            if &toks[start..start + n] == needle {
                let from = start + n;
                let to = (from + gamma).min(len);
                return (n, toks[from..to].to_vec());
            }
        }
    }
    (0, Vec::new())
}

fn synthetic_bytelevel_bpe_json() -> Vec<u8> {
    let mut bs: Vec<u32> = (u32::from(b'!')..=u32::from(b'~'))
        .chain(0xa1u32..=0xac)
        .chain(0xaeu32..=0xff)
        .collect();
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut vocab = serde_json::Map::new();
    vocab.insert("<|im_start|>".to_string(), serde_json::json!(0));
    vocab.insert("<|im_end|>".to_string(), serde_json::json!(1));
    let mut next_id = 2u32;
    for code in cs {
        let ch = char::from_u32(code).unwrap();
        vocab.insert(ch.to_string(), serde_json::json!(next_id));
        next_id += 1;
    }
    for tok in [
        "fn", "re", "et", "ret", "ur", "rn", "retu", "return", "in", "nt", "int",
    ] {
        vocab.insert(tok.to_string(), serde_json::json!(next_id));
        next_id += 1;
    }

    let doc = serde_json::json!({
        "version": "1.0",
        "added_tokens": [
            {"id": 0, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 1, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": { "type": "NFC" },
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": { "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" },
                    "behavior": "Isolated",
                    "invert": false
                },
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
            ]
        },
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": ["f n", "r e", "e t", "re t", "u r", "r n", "ret u", "retu rn", "i n", "n t", "in t"]
        }
    });
    serde_json::to_vec(&doc).unwrap()
}

fn discover_hf_tokenizers() -> Vec<(String, PathBuf)> {
    let mut bases = Vec::new();
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        bases.push(PathBuf::from(hf_home).join("hub"));
    }
    if let Ok(home) = std::env::var("HOME") {
        bases.push(
            PathBuf::from(home)
                .join(".cache")
                .join("huggingface")
                .join("hub"),
        );
    }

    let targets = [
        ("Qwen3-8B", "models--Qwen--Qwen3-8B"),
        ("Kimi-K2.5", "models--hoangquan456--Kimi-K2.5"),
        ("gpt-oss-120b", "models--openai--gpt-oss-120b"),
        ("GLM-4.7", "models--zai-org--GLM-4.7"),
        (
            "Nemotron-3-Nano-30B",
            "models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        ),
        ("DeepSeek-V3.2", "models--deepseek-ai--DeepSeek-V3.2"),
    ];

    let mut out = Vec::new();
    for (label, dir) in targets {
        for base in &bases {
            let snaps = base.join(dir).join("snapshots");
            if let Ok(entries) = std::fs::read_dir(&snaps) {
                for entry in entries.flatten() {
                    let p = entry.path().join("tokenizer.json");
                    if p.is_file() {
                        out.push((label.to_string(), p));
                        break;
                    }
                }
            }
            if out.last().is_some_and(|(n, _)| n == label) {
                break;
            }
        }
    }
    out
}

#[derive(Serialize)]
struct TokenizerBenchRow {
    model: String,
    hf_single_us: f64,
    hanzo_single_us: f64,
    speedup_vs_hf: f64,
    hanzo_20t_cold_us: f64,
    hanzo_20t_cached_us: f64,
    cache_hits: u64,
    cache_bypassed: u64,
}

#[derive(Serialize)]
struct SamBenchRow {
    ctx_len: usize,
    hit_naive_us: f64,
    hit_sam_us: f64,
    miss_naive_us: f64,
    miss_sam_us: f64,
}

#[derive(Serialize)]
struct VisionBenchRow {
    shape: String,
    baseline_ms: f64,
    simd_ms: f64,
    speedup: f64,
    max_abs_diff: f32,
}

#[derive(Serialize)]
struct HotpathReport {
    tokenizer: Vec<TokenizerBenchRow>,
    sam: Vec<SamBenchRow>,
    sa_spec_bypassed_rounds: usize,
    sa_spec_total_rounds: usize,
    chat_template_uncached_us: f64,
    chat_template_cached_us: f64,
    vision: Vec<VisionBenchRow>,
    kv_dir_512_p50_us: f64,
    kv_tcp_256k_push_us: f64,
    kv_tcp_256k_pull_us: f64,
}

fn bench_tokenizer_pair(
    label: &str,
    hf_tok: &HfTokenizer,
    fast_tok: &FastTokenizer,
) -> Result<TokenizerBenchRow> {
    let unit = "def compute_norm(xs: list[float]) -> float:\n    # café naïve 東京 🚀\n    return sum(x * x for x in xs)\n";
    let large_prompt = unit.repeat(320); // ~33 KiB

    let hf_ids = hf_tok.encode_fast(large_prompt.as_str(), true).unwrap();
    let hz_ids = fast_tok.encode_with_special_tokens(&large_prompt, true)?;
    anyhow::ensure!(
        hf_ids.get_ids() == hz_ids.as_slice(),
        "{label}: token parity mismatch on 33 KiB prompt"
    );

    let iters = 15;
    let mut hf_times = Vec::with_capacity(iters);
    let mut hz_times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let e = hf_tok.encode_fast(large_prompt.as_str(), true).unwrap();
        hf_times.push(t0.elapsed());
        std::hint::black_box(e);

        let t1 = Instant::now();
        let ids = fast_tok.encode_with_special_tokens(&large_prompt, true)?;
        hz_times.push(t1.elapsed());
        std::hint::black_box(ids);
    }
    let hf_p50 = percentile_us(&mut hf_times, 0.50);
    let hz_p50 = percentile_us(&mut hz_times, 0.50);

    // 20-turn incremental multi-turn benchmark
    let mut turns = Vec::with_capacity(20);
    let mut conv = String::from("<|im_start|>system\nYou are an assistant.<|im_end|>\n");
    for t in 0..20 {
        conv.push_str(&format!(
            "<|im_start|>user\nRefactor fn step_{t}(x: i32) -> i32 {{ x + {t} }}<|im_end|>\n\
             <|im_start|>assistant\n```rust\nfn step_{t}(x: i32) -> i32 {{ x + {t} }}\n```<|im_end|>\n"
        ));
        turns.push(conv.clone());
    }
    // Verify 20-turn PromptTokenCache parity against HF outside timed loops.
    let mut parity_cache = PromptTokenCache::new(16);
    for p in &turns {
        let got = fast_tok.encode_with_cache(p, true, &mut parity_cache)?;
        let exp = hf_tok.encode_fast(p.as_str(), true).unwrap();
        anyhow::ensure!(
            got.as_slice() == exp.get_ids(),
            "{label}: 20-turn PromptTokenCache parity mismatch"
        );
    }

    // Warmup cold path before measuring median 20-turn cold vs cached latency.
    for p in &turns {
        std::hint::black_box(fast_tok.encode_with_special_tokens(p, true)?);
    }

    let mut cold_samples = Vec::with_capacity(5);
    for _ in 0..5 {
        let t_cold = Instant::now();
        for p in &turns {
            std::hint::black_box(fast_tok.encode_with_special_tokens(p, true)?);
        }
        cold_samples.push(t_cold.elapsed().as_secs_f64() * 1_000_000.0);
    }
    cold_samples.sort_by(f64::total_cmp);
    let cold_20t_us = cold_samples[cold_samples.len() / 2];

    let mut cached_samples = Vec::with_capacity(5);
    let mut stats = parity_cache.stats();
    for _ in 0..5 {
        let mut cache = PromptTokenCache::new(16);
        let t_cached = Instant::now();
        for p in &turns {
            std::hint::black_box(fast_tok.encode_with_cache(p, true, &mut cache)?);
        }
        cached_samples.push(t_cached.elapsed().as_secs_f64() * 1_000_000.0);
        stats = cache.stats();
    }
    cached_samples.sort_by(f64::total_cmp);
    let cached_20t_us = cached_samples[cached_samples.len() / 2];

    println!(
        "  tokenizer [{label:<20}]: hf={hf_p50:>8.1}us hanzo={hz_p50:>7.1}us ({:.2}x) | 20T cold={cold_20t_us:>7.1}us cached={cached_20t_us:>7.1}us (hits={} bypassed={})",
        hf_p50 / hz_p50.max(1e-6),
        stats.exact_hits + stats.prefix_hits,
        stats.bypassed
    );

    Ok(TokenizerBenchRow {
        model: label.to_string(),
        hf_single_us: hf_p50,
        hanzo_single_us: hz_p50,
        speedup_vs_hf: hf_p50 / hz_p50.max(1e-6),
        hanzo_20t_cold_us: cold_20t_us,
        hanzo_20t_cached_us: cached_20t_us,
        cache_hits: stats.exact_hits + stats.prefix_hits,
        cache_bypassed: stats.bypassed,
    })
}

fn bench_speculative_sam() -> Result<(Vec<SamBenchRow>, usize, usize)> {
    let mut rows = Vec::new();
    for &ctx_len in &[1024usize, 4096, 16384] {
        let mut hit_seq: Vec<u32> = (0..ctx_len as u32).map(|i| (i % 97) + 100).collect();
        let needle: Vec<u32> = (0..8u32).map(|i| 10 + i).collect();
        let mid = ctx_len / 2;
        hit_seq[mid..mid + 8].copy_from_slice(&needle);
        hit_seq.extend_from_slice(&needle);

        let mut miss_seq: Vec<u32> = (0..ctx_len as u32).map(|i| i + 100).collect();
        miss_seq.extend_from_slice(&[999_991, 999_992, 999_993, 999_994]);

        let mut sam_hit = hanzo_engine::speculative::SuffixAutomaton::new();
        let mut sam_miss = hanzo_engine::speculative::SuffixAutomaton::new();
        let _ = sam_hit.lookup(&hit_seq, 2, 8, 6);
        let _ = sam_miss.lookup(&miss_seq, 2, 8, 6);

        let iters = 30;
        let mut h_naive = Vec::with_capacity(iters);
        let mut h_sam = Vec::with_capacity(iters);
        let mut m_naive = Vec::with_capacity(iters);
        let mut m_sam = Vec::with_capacity(iters);

        for k in 0..iters {
            let tok = 5000 + k as u32;
            hit_seq.push(tok);
            let t0 = Instant::now();
            let n_res = naive_lookup(&hit_seq, 2, 8, 6);
            h_naive.push(t0.elapsed());

            let t1 = Instant::now();
            let s_res = sam_hit.lookup(&hit_seq, 2, 8, 6);
            h_sam.push(t1.elapsed());
            anyhow::ensure!(n_res == s_res, "SAM hit parity mismatch at len={ctx_len}");

            miss_seq.push(1_000_000 + k as u32);
            let t2 = Instant::now();
            let mn = naive_lookup(&miss_seq, 2, 8, 6);
            m_naive.push(t2.elapsed());

            let t3 = Instant::now();
            let ms = sam_miss.lookup(&miss_seq, 2, 8, 6);
            m_sam.push(t3.elapsed());
            anyhow::ensure!(mn == ms, "SAM miss parity mismatch at len={ctx_len}");
        }

        let row = SamBenchRow {
            ctx_len,
            hit_naive_us: percentile_us(&mut h_naive, 0.50),
            hit_sam_us: percentile_us(&mut h_sam, 0.50),
            miss_naive_us: percentile_us(&mut m_naive, 0.50),
            miss_sam_us: percentile_us(&mut m_sam, 0.50),
        };
        println!(
            "  SAM [ctx={:>5}]: hit naive={:.2}us sam={:.2}us | miss naive={:.2}us sam={:.2}us ({:.1}x miss speedup)",
            row.ctx_len,
            row.hit_naive_us,
            row.hit_sam_us,
            row.miss_naive_us,
            row.miss_sam_us,
            row.miss_naive_us / row.miss_sam_us.max(1e-6)
        );
        rows.push(row);
    }

    // 64-step grounded edit decode SA_SPEC bypass check
    let base: Vec<u32> = (0..256u32).cycle().take(8192).collect();
    let mut seq = base.clone();
    seq.extend_from_slice(&base[400..406]);
    let mut sam = hanzo_engine::speculative::SuffixAutomaton::new();
    let mut bypassed = 0usize;
    let total_steps = 64usize;
    for step in 0..total_steps {
        let (matched_len, prop) = sam.lookup(&seq, 2, 8, 6);
        if matched_len >= 4 && !prop.is_empty() {
            bypassed += 1;
        }
        seq.push(base[406 + step]);
    }
    println!(
        "  SA_SPEC grounded edit decode: {bypassed}/{total_steps} neural draft passes bypassed"
    );
    Ok((rows, bypassed, total_steps))
}

fn bench_chat_template() -> Result<(f64, f64)> {
    let tmpl_src = r#"{%- if tools -%}
<|im_start|>system
Tools:
{%- for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{%- endfor %}<|im_end|>
{%- endif -%}
{%- for m in messages %}
<|im_start|>{{ m.role }}
{{ m.content | trim }}<|im_end|>
{%- endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{% endif %}"#;

    let mut messages = Vec::new();
    for i in 0..12 {
        messages.push(HashMap::from([
            (
                "role",
                if i % 2 == 0 { "user" } else { "assistant" }.to_string(),
            ),
            (
                "content",
                format!("Turn {i}: update module src/engine_{i}.rs and verify tests."),
            ),
        ]));
    }
    let tools = [
        HashMap::from([
            ("name", "read_file"),
            ("description", "Read workspace file lines"),
        ]),
        HashMap::from([
            ("name", "edit_file"),
            ("description", "Apply surgical edit to file"),
        ]),
    ];

    let mut cached_env = Environment::new();
    cached_env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    cached_env.add_template("chat", tmpl_src)?;

    let mut uncached_times = Vec::with_capacity(50);
    let mut cached_times = Vec::with_capacity(50);
    for _ in 0..50 {
        let t0 = Instant::now();
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template("chat", tmpl_src)?;
        let out0 = env.get_template("chat")?.render(
            context! { messages => &messages, tools => &tools, add_generation_prompt => true },
        )?;
        uncached_times.push(t0.elapsed());

        let t1 = Instant::now();
        let out1 = cached_env.get_template("chat")?.render(
            context! { messages => &messages, tools => &tools, add_generation_prompt => true },
        )?;
        cached_times.push(t1.elapsed());
        anyhow::ensure!(out0 == out1, "chat template output parity mismatch");
    }
    let uncached_us = percentile_us(&mut uncached_times, 0.50);
    let cached_us = percentile_us(&mut cached_times, 0.50);
    println!(
        "  chat_template (12T+2tools): uncached={uncached_us:.1}us | cached={cached_us:.1}us ({:.2}x)",
        uncached_us / cached_us.max(1e-6)
    );
    Ok((uncached_us, cached_us))
}

fn bench_vision() -> Result<Vec<VisionBenchRow>> {
    let mut rows = Vec::new();
    let dev = Device::Cpu;
    let baseline_tf = Transforms {
        input: &ToTensor,
        inner_transforms: &[&Normalize {
            mean: vec![0.5, 0.5, 0.5],
            std: vec![0.5, 0.5, 0.5],
        }],
    };
    let fast_tf = ToNormalizedTensor {
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
    };

    for (label, w, h) in [("1080p", 1920u32, 1080u32), ("4K", 3840u32, 2160u32)] {
        let mut raw = vec![0u8; (w * h * 3) as usize];
        for (i, b) in raw.iter_mut().enumerate() {
            *b = ((i * 31) % 251) as u8;
        }
        let img = DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, raw).unwrap());

        let base_t = img
            .resize_exact(896, 896, FilterType::CatmullRom)
            .apply(baseline_tf, &dev)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let fast_img = simd_resize_rgb(&img, 896, 896)?;
        let ref_norm_on_fast = fast_img
            .clone()
            .apply(baseline_tf, &dev)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let fast_t = fast_tf
            .map(&fast_img, &dev)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let max_diff = ref_norm_on_fast
            .iter()
            .zip(fast_t.iter())
            .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));
        anyhow::ensure!(
            max_diff < 1e-5,
            "vision normalization drift too high on {label}: {max_diff}"
        );
        let resize_diff = base_t
            .iter()
            .zip(fast_t.iter())
            .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));
        anyhow::ensure!(
            resize_diff <= (1.0 / 127.5) + 1e-4,
            "vision resize+norm drift exceeds 1 u8 LSB on {label}: {resize_diff}"
        );

        let iters = 8;
        let mut base_times = Vec::with_capacity(iters);
        let mut fast_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let b_out = img
                .resize_exact(896, 896, FilterType::CatmullRom)
                .apply(baseline_tf, &dev)?;
            base_times.push(t0.elapsed());
            std::hint::black_box(b_out);

            let t1 = Instant::now();
            let r = simd_resize_rgb(&img, 896, 896)?;
            let f_out = fast_tf.map(&r, &dev)?;
            fast_times.push(t1.elapsed());
            std::hint::black_box(f_out);
        }
        let b_ms = percentile_ms(&mut base_times, 0.50);
        let f_ms = percentile_ms(&mut fast_times, 0.50);
        let spd = b_ms / f_ms.max(1e-6);
        println!(
            "  vision [{label} -> 896x896 CHW]: baseline={b_ms:.2}ms | hanzo_simd={f_ms:.2}ms ({spd:.2}x, max_diff={max_diff:.2e})"
        );
        rows.push(VisionBenchRow {
            shape: format!("{w}x{h}->896x896"),
            baseline_ms: b_ms,
            simd_ms: f_ms,
            speedup: spd,
            max_abs_diff: max_diff,
        });
    }
    Ok(rows)
}

async fn bench_kv_fabric() -> Result<(f64, f64, f64)> {
    let dir = EpochFencedKvDirectory::new();
    let backend = Arc::new(InMemoryKvBackend::new(1));
    let transport = KvBlockTransport::bind("127.0.0.1:0".parse()?, Arc::clone(&backend)).await?;
    let peer = transport.local_addr();

    let digest = compute_block_digest(b"bench-meta");
    let metas: Vec<KvBlockMeta> = (0u32..512)
        .map(|i| KvBlockMeta {
            block_hash: kv_block_hash(&[i, i + 1], 0),
            num_bytes: 262_144,
            sha256: digest,
        })
        .collect();
    let mut dir_times = Vec::with_capacity(30);
    for epoch in 1u64..=30 {
        let t0 = Instant::now();
        dir.publish_blocks("worker-0", epoch, peer, &metas)?;
        let _ = dir.lookup_fresh_owners(metas[256].block_hash);
        let _ = dir.fence_epoch("worker-0", epoch + 1);
        dir_times.push(t0.elapsed());
    }
    let dir_p50 = percentile_us(&mut dir_times, 0.50);

    backend.set_epoch(100);
    let payload = vec![0x5Au8; 256 * 1024];
    let hash = kv_block_hash(&[7, 256], 0);
    let mut push_times = Vec::with_capacity(20);
    let mut pull_times = Vec::with_capacity(20);
    for i in 0..20 {
        let ep = 100 + i as u64;
        let t0 = Instant::now();
        KvBlockTransport::<InMemoryKvBackend>::push_block(peer, hash, ep, &payload).await?;
        push_times.push(t0.elapsed());
        let t1 = Instant::now();
        let (_, got) = KvBlockTransport::<InMemoryKvBackend>::pull_block(peer, hash, ep).await?;
        pull_times.push(t1.elapsed());
        anyhow::ensure!(got.len() == payload.len());
    }
    let push_p50 = percentile_us(&mut push_times, 0.50);
    let pull_p50 = percentile_us(&mut pull_times, 0.50);
    println!(
        "  kv_fabric: dir512={dir_p50:.1}us | HKV1 256KiB push={push_p50:.1}us pull={pull_p50:.1}us"
    );
    Ok((dir_p50, push_p50, pull_p50))
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let json_out = args
        .windows(2)
        .find(|w| w[0] == "--json")
        .map(|w| PathBuf::from(&w[1]));

    println!("=== Hanzo Hot-Path CPU Release Benchmark ===");

    let mut tok_rows = Vec::new();
    let synth_bytes = synthetic_bytelevel_bpe_json();
    let synth_hf = HfTokenizer::from_bytes(&synth_bytes).unwrap();
    let synth_fast = FastTokenizer::from_bytes(&synth_bytes)?;
    tok_rows.push(bench_tokenizer_pair(
        "synthetic-cl100k",
        &synth_hf,
        &synth_fast,
    )?);

    for (label, path) in discover_hf_tokenizers() {
        let hf_tok = HfTokenizer::from_file(&path).unwrap();
        let fast_tok = FastTokenizer::from_file(&path)?;
        tok_rows.push(bench_tokenizer_pair(&label, &hf_tok, &fast_tok)?);
    }

    let (sam_rows, sa_bypassed, sa_total) = bench_speculative_sam()?;
    let (chat_uncached, chat_cached) = bench_chat_template()?;
    let vision_rows = bench_vision()?;
    let (dir_p50, push_p50, pull_p50) = bench_kv_fabric().await?;

    if let Some(path) = json_out {
        let report = HotpathReport {
            tokenizer: tok_rows,
            sam: sam_rows,
            sa_spec_bypassed_rounds: sa_bypassed,
            sa_spec_total_rounds: sa_total,
            chat_template_uncached_us: chat_uncached,
            chat_template_cached_us: chat_cached,
            vision: vision_rows,
            kv_dir_512_p50_us: dir_p50,
            kv_tcp_256k_push_us: push_p50,
            kv_tcp_256k_pull_us: pull_p50,
        };
        std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;
        println!("wrote report to {}", path.display());
    }

    Ok(())
}
