//! Accelerated tokenization pool using [`hanzo_tokenizer`] with boundary-safe
//! multi-turn [`PromptTokenCache`] prefix reuse, gated behind an expanded
//! 7-probe parity self-check against Hugging Face `tokenizers`.

use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock, Weak};

use anyhow::Result;
use hanzo_tokenizer::{PromptTokenCache, Tokenizer as FastTokenizer};
use indexmap::IndexMap;
use parking_lot::{Mutex, RwLock};
use tokenizers::Tokenizer as HfTokenizer;

const FAST_POOL_CAP: usize = 16;
const PROMPT_CACHE_CAP: usize = 16;

const STATIC_PARITY_PROBES: [&str; 4] = [
    "def fib(n: int) -> int:\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
    "Unicode NFC/NFD: cafe\u{0301} café naïve 東京 العربية 👨‍👩‍👧‍👦 🚀 — 42 != 0\n",
    "\r\n\t  /home/dev/src/main_v2.rs:128:4   foo_bar_99\r\n\r\n",
    "{\"tool\":\"read_file\",\"args\":{\"path\":\"src/lib.rs\",\"lines\":[1,120]}}\n",
];

enum FastBackend {
    Accelerated {
        tokenizer: FastTokenizer,
        cache: Mutex<PromptTokenCache>,
    },
    Fallback,
}

struct CachedFastTokenizer {
    origin: Weak<HfTokenizer>,
    backend: Arc<FastBackend>,
}

static FAST_TOKENIZER_POOL: LazyLock<RwLock<IndexMap<usize, CachedFastTokenizer>>> =
    LazyLock::new(|| RwLock::new(IndexMap::with_capacity(FAST_POOL_CAP)));

fn check_pair(
    hf_tok: &HfTokenizer,
    fast_tok: &FastTokenizer,
    text: &str,
    add_spec: bool,
    cache: Option<&mut PromptTokenCache>,
) -> bool {
    let Ok(hf_enc) = hf_tok.encode_fast(text, add_spec) else {
        return false;
    };
    let fast_ids = match cache {
        Some(c) => fast_tok.encode_with_cache(text, add_spec, c),
        None => fast_tok.encode_with_special_tokens(text, add_spec),
    };
    let Ok(fast_ids) = fast_ids else {
        return false;
    };
    if hf_enc.get_ids() != fast_ids.as_slice() {
        tracing::debug!(
            add_special_tokens = add_spec,
            hf_len = hf_enc.get_ids().len(),
            fast_len = fast_ids.len(),
            "FastTokenizerPool probe mismatch; falling back to tokenizers"
        );
        return false;
    }
    true
}

fn dynamic_added_token_probe(hf_tok: &HfTokenizer) -> Option<String> {
    let decoded = hf_tok.get_added_tokens_decoder();
    if decoded.is_empty() {
        return None;
    }
    let ordered: BTreeMap<u32, _> = decoded.into_iter().collect();
    let mut specials: Vec<String> = Vec::with_capacity(4);
    let mut non_specials: Vec<String> = Vec::with_capacity(4);
    for tok in ordered.into_values() {
        if tok.content.is_empty() {
            continue;
        }
        if tok.special {
            if specials.len() < 4 {
                specials.push(tok.content);
            }
        } else if non_specials.len() < 4 {
            non_specials.push(tok.content);
        }
        if specials.len() == 4 && non_specials.len() == 4 {
            break;
        }
    }
    if specials.is_empty() && non_specials.is_empty() {
        return None;
    }
    let s0 = specials.first().map(String::as_str).unwrap_or("");
    let s1 = specials
        .get(1)
        .or_else(|| specials.first())
        .map(String::as_str)
        .unwrap_or("");
    let ns0 = non_specials.first().map(String::as_str).unwrap_or("");
    let ns1 = non_specials
        .get(1)
        .or_else(|| non_specials.first())
        .map(String::as_str)
        .unwrap_or("");
    Some(format!(
        "{s0}system\nYou are helpful.{s1}\n{s0}user\nTest {ns0} {ns1} token edges {ns0}.{s1}\n"
    ))
}

fn build_fast_backend(hf_tok: &HfTokenizer) -> FastBackend {
    let Ok(json_str) = hf_tok.to_string(false) else {
        return FastBackend::Fallback;
    };
    let Ok(fast_tok) = FastTokenizer::from_bytes(json_str.as_bytes()) else {
        return FastBackend::Fallback;
    };

    // Probes 1..=4: ASCII/code, Unicode NFC+NFD+CJK+RTL+ZWJ, CRLF/tabs/paths, JSON tool framing.
    for &probe in &STATIC_PARITY_PROBES {
        for &add_spec in &[false, true] {
            if !check_pair(hf_tok, &fast_tok, probe, add_spec, None) {
                return FastBackend::Fallback;
            }
        }
    }

    // Probe 5: dynamic special and non-special added-token boundary probe.
    if let Some(added_probe) = dynamic_added_token_probe(hf_tok) {
        for &add_spec in &[false, true] {
            if !check_pair(hf_tok, &fast_tok, &added_probe, add_spec, None) {
                return FastBackend::Fallback;
            }
        }
    }

    // Probes 6..=7: 2-turn incremental PromptTokenCache parity across both special-token modes.
    let turn1 = format!(
        "System: verify incremental cache safety boundaries across lines and checkpoints.\n{}",
        "Line alpha_beta_0123456789 path/to/src/lib.rs -> ok\n".repeat(12)
    );
    let turn2 =
        format!("{turn1}User turn 2: append résumé naïve 東京 🚀 with slash .\n/tmp/check\n");
    for &add_spec in &[false, true] {
        let mut probe_cache = PromptTokenCache::new(4);
        if !check_pair(hf_tok, &fast_tok, &turn1, add_spec, Some(&mut probe_cache))
            || !check_pair(hf_tok, &fast_tok, &turn2, add_spec, Some(&mut probe_cache))
        {
            return FastBackend::Fallback;
        }
    }

    FastBackend::Accelerated {
        tokenizer: fast_tok,
        cache: Mutex::new(PromptTokenCache::new(PROMPT_CACHE_CAP)),
    }
}

fn get_or_init_backend(hf_tok: &Arc<HfTokenizer>) -> Arc<FastBackend> {
    let key = Arc::as_ptr(hf_tok) as usize;
    let origin = Arc::downgrade(hf_tok);
    {
        let pool = FAST_TOKENIZER_POOL.read();
        if let Some(entry) = pool.get(&key) {
            if entry.origin.ptr_eq(&origin) {
                return Arc::clone(&entry.backend);
            }
        }
    }

    // Build and verify outside the lock so concurrent tokenizations never block.
    let built = Arc::new(build_fast_backend(hf_tok));

    let mut pool = FAST_TOKENIZER_POOL.write();
    if let Some(entry) = pool.get(&key) {
        if entry.origin.ptr_eq(&origin) {
            return Arc::clone(&entry.backend);
        }
    }

    pool.retain(|_, entry| entry.origin.strong_count() > 0);
    while pool.len() >= FAST_POOL_CAP {
        pool.shift_remove_index(0);
    }
    pool.insert(
        key,
        CachedFastTokenizer {
            origin,
            backend: Arc::clone(&built),
        },
    );
    built
}

/// Encode `text` using [`hanzo_tokenizer`] + [`PromptTokenCache`] when the live
/// 7-probe parity gate passes for `hf_tok`, falling back to `hf_tok.encode_fast`
/// otherwise.
pub(crate) fn encode_prompt_cached(
    hf_tok: &Arc<HfTokenizer>,
    text: &str,
    add_special_tokens: bool,
) -> Result<Vec<u32>> {
    let backend = get_or_init_backend(hf_tok);
    match backend.as_ref() {
        FastBackend::Accelerated { tokenizer, cache } => {
            if let Some(mut guard) = cache.try_lock() {
                tokenizer
                    .encode_with_cache(text, add_special_tokens, &mut guard)
                    .map_err(anyhow::Error::msg)
            } else {
                tokenizer
                    .encode_with_special_tokens(text, add_special_tokens)
                    .map_err(anyhow::Error::msg)
            }
        }
        FastBackend::Fallback => hf_tok
            .encode_fast(text, add_special_tokens)
            .map(|enc| enc.get_ids().to_vec())
            .map_err(anyhow::Error::msg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn bytelevel_256_vocab() -> (serde_json::Map<String, serde_json::Value>, u32) {
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
        vocab.insert("<|endoftext|>".to_string(), json!(0));
        let mut next_id = 1u32;
        for code in cs {
            let ch = char::from_u32(code).unwrap();
            vocab.insert(ch.to_string(), json!(next_id));
            next_id += 1;
        }
        (vocab, next_id)
    }

    fn bytelevel_bpe_test_tokenizer() -> Arc<HfTokenizer> {
        let (mut vocab, mut next_id) = bytelevel_256_vocab();
        let merged_tokens = [
            "de", "ef", "def", "re", "et", "ret", "ur", "rn", "retu", "return", "in", "nt", "int",
            "fn", "Ġf", "Ġi", "Ġr", "Ġt", "Ġthe", "th", "the", "ca", "af", "caf",
        ];
        for tok in merged_tokens {
            vocab.insert(tok.to_string(), json!(next_id));
            next_id += 1;
        }
        let merges = [
            "d e", "d e f", "de f", "r e", "t u", "r n", "re t", "u r", "ret u", "retu rn", "i n",
            "in t", "f n", "Ġ f", "Ġ i", "Ġ r", "Ġ t", "t h", "th e", "Ġ t h", "Ġt h", "Ġ th",
            "Ġthe", "c a", "a f", "ca f",
        ];
        // Keep only pairs whose inputs exist in `vocab`.
        let valid_merges: Vec<&str> = merges
            .into_iter()
            .filter(|m| {
                let Some((a, b)) = m.split_once(' ') else {
                    return false;
                };
                vocab.contains_key(a)
                    && vocab.contains_key(b)
                    && vocab.contains_key(&format!("{a}{b}"))
            })
            .collect();

        let json_val = json!({
            "version": "1.0",
            "added_tokens": [
                {
                    "id": 0,
                    "content": "<|endoftext|>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                }
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
                    {
                        "type": "ByteLevel",
                        "add_prefix_space": false,
                        "trim_offsets": true,
                        "use_regex": false
                    }
                ]
            },
            "post_processor": {
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": true,
                "use_regex": true
            },
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": true,
                "use_regex": true
            },
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": valid_merges
            }
        });

        let tok = HfTokenizer::from_bytes(serde_json::to_vec(&json_val).unwrap()).unwrap();
        Arc::new(tok)
    }

    fn o200k_template_bpe_test_tokenizer() -> Arc<HfTokenizer> {
        let (mut vocab, mut next_id) = bytelevel_256_vocab();
        for tok in [
            "<|im_start|>",
            "<|im_end|>",
            "<bos>",
            "fn",
            "Ġr",
            "ru",
            "un",
            "run",
        ] {
            vocab.insert(tok.to_string(), json!(next_id));
            next_id += 1;
        }
        let im_start_id = vocab["<|im_start|>"].clone();
        let im_end_id = vocab["<|im_end|>"].clone();
        let bos_id = vocab["<bos>"].clone();

        let json_val = json!({
            "version": "1.0",
            "added_tokens": [
                {"id": im_start_id, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": im_end_id, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": bos_id, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": { "type": "NFC" },
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": { "Regex": "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" },
                        "behavior": "Isolated",
                        "invert": false
                    },
                    {
                        "type": "ByteLevel",
                        "add_prefix_space": false,
                        "trim_offsets": true,
                        "use_regex": false
                    }
                ]
            },
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                    { "SpecialToken": { "id": "<bos>", "type_id": 0 } },
                    { "Sequence": { "id": "A", "type_id": 0 } }
                ],
                "pair": [
                    { "SpecialToken": { "id": "<bos>", "type_id": 0 } },
                    { "Sequence": { "id": "A", "type_id": 0 } },
                    { "Sequence": { "id": "B", "type_id": 1 } }
                ],
                "special_tokens": {
                    "<bos>": { "id": "<bos>", "ids": [bos_id], "tokens": ["<bos>"] }
                }
            },
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": ["f n", "r u", "ru n", "Ġ r"]
            }
        });
        Arc::new(HfTokenizer::from_bytes(serde_json::to_vec(&json_val).unwrap()).unwrap())
    }

    fn discover_cached_hf_tokenizers() -> Vec<(&'static str, std::path::PathBuf)> {
        let mut bases = Vec::new();
        if let Ok(hf_home) = std::env::var("HF_HOME") {
            bases.push(std::path::PathBuf::from(hf_home).join("hub"));
        }
        if let Ok(home) = std::env::var("HOME") {
            bases.push(
                std::path::PathBuf::from(home)
                    .join(".cache")
                    .join("huggingface")
                    .join("hub"),
            );
        }

        let specs = [
            ("qwen3-8b", "models--Qwen--Qwen3-8B"),
            ("kimi-k2.5", "models--hoangquan456--Kimi-K2.5"),
            ("gpt-oss-120b", "models--openai--gpt-oss-120b"),
            ("glm-4.7", "models--zai-org--GLM-4.7"),
            (
                "nemotron-3-nano-30b",
                "models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            ),
            ("deepseek-v3.2", "models--deepseek-ai--DeepSeek-V3.2"),
        ];

        let mut found = Vec::new();
        for (name, repo_dir) in specs {
            for base in &bases {
                let snapshots = base.join(repo_dir).join("snapshots");
                if let Ok(entries) = std::fs::read_dir(&snapshots) {
                    for entry in entries.flatten() {
                        let tok = entry.path().join("tokenizer.json");
                        if tok.is_file() {
                            found.push((name, tok));
                            break;
                        }
                    }
                }
                if found.last().is_some_and(|(n, _)| *n == name) {
                    break;
                }
            }
        }
        found
    }

    #[test]
    fn fast_pool_accelerates_bytelevel_bpe_and_reuses_prefix_cache() {
        let hf_tok = bytelevel_bpe_test_tokenizer();
        let backend = get_or_init_backend(&hf_tok);
        assert!(
            matches!(backend.as_ref(), FastBackend::Accelerated { .. }),
            "Qwen2-style NFC + Split + ByteLevel should pass 3-probe parity check"
        );

        let mut prompt = String::from("System prompt: assist accurately.\n");
        for turn in 0..8 {
            prompt.push_str(&format!(
                "User turn {turn}: summarize café naïve 🚀 item #{turn}\nAssistant turn {turn}: done.\n"
            ));
            for &add_spec in &[false, true] {
                let got = encode_prompt_cached(&hf_tok, &prompt, add_spec).unwrap();
                let expected = hf_tok.encode_fast(prompt.as_str(), add_spec).unwrap();
                assert_eq!(got.as_slice(), expected.get_ids());
            }
        }

        // Repeat exact prompt once so exact_hits > 0 deterministically alongside prefix_hits > 0.
        let repeated = encode_prompt_cached(&hf_tok, &prompt, false).unwrap();
        let expected_rep = hf_tok.encode_fast(prompt.as_str(), false).unwrap();
        assert_eq!(repeated.as_slice(), expected_rep.get_ids());

        if let FastBackend::Accelerated { cache, .. } = backend.as_ref() {
            let stats = cache.lock().stats();
            assert!(
                stats.exact_hits > 0 && stats.prefix_hits > 0,
                "expected both exact and prefix cache hits, got {stats:?}"
            );
        }
    }

    #[test]
    fn fast_pool_handles_fallback_and_stale_weak_pointers_safely() {
        // Unsupported WordLevel fallback still returns exact HF IDs.
        let wl_json = json!({
            "version": "1.0",
            "added_tokens": [],
            "normalizer": { "type": "Lowercase" },
            "pre_tokenizer": { "type": "WhitespaceSplit" },
            "model": {
                "type": "WordLevel",
                "vocab": { "[UNK]": 0, "hello": 1, "world": 2 },
                "unk_token": "[UNK]"
            }
        });
        let hf_wl =
            Arc::new(HfTokenizer::from_bytes(serde_json::to_vec(&wl_json).unwrap()).unwrap());
        let got = encode_prompt_cached(&hf_wl, "Hello world", false).unwrap();
        assert_eq!(got, vec![1, 2]);

        // Dropping an Arc removes its stale Weak entry on next insert without ABA aliasing.
        let ephemeral = bytelevel_bpe_test_tokenizer();
        let _ = encode_prompt_cached(&ephemeral, "ephemeral check\n", false).unwrap();
        drop(ephemeral);

        let replacement = bytelevel_bpe_test_tokenizer();
        let got2 = encode_prompt_cached(&replacement, "replacement check\n", false).unwrap();
        let expected2 = replacement
            .encode_fast("replacement check\n", false)
            .unwrap();
        assert_eq!(got2.as_slice(), expected2.get_ids());
    }

    #[test]
    fn fast_pool_unicode_nfc_nfd_zwj_and_whitespace_parity() {
        let fixtures = [
            bytelevel_bpe_test_tokenizer(),
            o200k_template_bpe_test_tokenizer(),
        ];
        let samples = [
            "e\u{0301}clair cafe\u{0301} naïve résumé Ω≈ç√∫˜µ≤≥÷",
            "CJK + RTL: 日本語テスト 中文分词 العربية שלום",
            "Emoji ZWJ: 👨‍👩‍👧‍👦 👩🏽‍💻 🏳️‍🌈 🚀✨🔥",
            "Whitespace & CRLF:\r\n\t   /home/user/workspace/hanzo-engine/src/main.rs\r\n   \t ",
        ];
        for hf_tok in &fixtures {
            assert!(matches!(
                get_or_init_backend(hf_tok).as_ref(),
                FastBackend::Accelerated { .. }
            ));
            for sample in samples {
                for &add_spec in &[false, true] {
                    let got = encode_prompt_cached(hf_tok, sample, add_spec).unwrap();
                    let exp = hf_tok.encode_fast(sample, add_spec).unwrap();
                    assert_eq!(got.as_slice(), exp.get_ids(), "mismatch on {sample:?}");
                }
            }
        }
    }

    #[test]
    fn fast_pool_special_and_nonspecial_added_tokens_parity() {
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
        for (idx, code) in cs.iter().enumerate() {
            let ch = char::from_u32(*code).unwrap();
            vocab.insert(ch.to_string(), json!(idx));
        }
        vocab.insert("<|im_start|>".to_string(), json!(256));
        vocab.insert("<|im_end|>".to_string(), json!(257));
        vocab.insert("<tool_tag>".to_string(), json!(258));
        vocab.insert("CUSTOM_OP".to_string(), json!(259));

        let json_val = json!({
            "version": "1.0",
            "added_tokens": [
                {"id": 256, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 257, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 258, "content": "<tool_tag>", "single_word": false, "lstrip": true, "rstrip": true, "normalized": false, "special": false},
                {"id": 259, "content": "CUSTOM_OP", "single_word": true, "lstrip": false, "rstrip": false, "normalized": false, "special": false}
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
            "model": { "type": "BPE", "vocab": vocab, "merges": [] }
        });

        let hf_tok =
            Arc::new(HfTokenizer::from_bytes(serde_json::to_vec(&json_val).unwrap()).unwrap());
        let cases = [
            "<|im_start|>system\nhello  <tool_tag>  world<|im_end|>\n",
            "<|im_start|>user\nCUSTOM_OP in-word CUSTOM_OPx xCUSTOM_OP\n<|im_end|>\n",
        ];
        for case in cases {
            for &add_spec in &[false, true] {
                let got = encode_prompt_cached(&hf_tok, case, add_spec).unwrap();
                let exp = hf_tok.encode_fast(case, add_spec).unwrap();
                assert_eq!(
                    got.as_slice(),
                    exp.get_ids(),
                    "added-token mismatch for {case:?}"
                );
            }
        }
    }

    #[test]
    fn fast_pool_o200k_template_and_cached_hf_models_parity() {
        for (name, hf_tok) in [
            ("synthetic-cl100k", bytelevel_bpe_test_tokenizer()),
            (
                "synthetic-o200k-template",
                o200k_template_bpe_test_tokenizer(),
            ),
        ]
        .into_iter()
        .chain(
            discover_cached_hf_tokenizers()
                .into_iter()
                .map(|(name, path)| (name, Arc::new(HfTokenizer::from_file(path).unwrap()))),
        ) {
            let backend = get_or_init_backend(&hf_tok);
            assert!(
                matches!(backend.as_ref(), FastBackend::Accelerated { .. }),
                "{name} should pass 7-probe gate"
            );

            let mut conv = String::from("<|im_start|>system\nYou are Zen coder.\n<|im_end|>\n");
            for turn in 0..8 {
                conv.push_str(&format!(
                    "<|im_start|>user\nEdit fn run_{turn}() -> usize {{ {turn} }}\n<|im_end|>\n\
                     <|im_start|>assistant\n```rust\nfn run_{turn}() -> usize {{ {turn} }}\n```\n<|im_end|>\n"
                ));
                let got = encode_prompt_cached(&hf_tok, &conv, true).unwrap();
                let exp = hf_tok.encode_fast(conv.as_str(), true).unwrap();
                assert_eq!(
                    got.as_slice(),
                    exp.get_ids(),
                    "{name} turn {turn} parity failed"
                );
            }

            if let FastBackend::Accelerated { cache, .. } = backend.as_ref() {
                let inc_stats = cache.lock().stats();
                assert!(
                    inc_stats.prefix_hits > 0,
                    "{name} should reuse newline prefix cache on incremental stream, got {inc_stats:?}"
                );
            }

            // Verify self-bypass engages cleanly on a non-incremental miss stream without losing parity.
            for idx in 0..10 {
                let doc = format!(
                    "Unrelated prompt #{idx} payload line\nSecond line with unique token #{idx}\n"
                );
                let got = encode_prompt_cached(&hf_tok, &doc, true).unwrap();
                let exp = hf_tok.encode_fast(doc.as_str(), true).unwrap();
                assert_eq!(
                    got.as_slice(),
                    exp.get_ids(),
                    "{name} non-incremental #{idx} parity failed"
                );
            }

            if let FastBackend::Accelerated { cache, .. } = backend.as_ref() {
                let stats = cache.lock().stats();
                assert!(
                    stats.bypassed > 0,
                    "{name} should engage auto-bypass on non-incremental stream, got {stats:?}"
                );
            }

            let false_probe = format!(
                "{conv}<|im_start|>user\nCheck add_special_tokens=false parity.\n<|im_end|>\n"
            );
            let got_false = encode_prompt_cached(&hf_tok, &false_probe, false).unwrap();
            let exp_false = hf_tok.encode_fast(false_probe.as_str(), false).unwrap();
            assert_eq!(
                got_false.as_slice(),
                exp_false.get_ids(),
                "{name} add_special_tokens=false parity failed"
            );
        }
    }

    #[test]
    fn fast_pool_concurrent_16_threads_stress_parity() {
        let hf_tok = o200k_template_bpe_test_tokenizer();

        std::thread::scope(|s| {
            for tid in 0..16usize {
                let tok = Arc::clone(&hf_tok);
                s.spawn(move || {
                    let mut prompt = format!("Thread {tid} system init.\n");
                    for turn in 0..32usize {
                        prompt.push_str(&format!(
                            "T{tid} turn {turn}: café naïve 🚀 foo_{tid}_{turn}();\nNext line padding 123456.\n"
                        ));
                        let add_spec = turn % 2 == 0;
                        let got = encode_prompt_cached(&tok, &prompt, add_spec).unwrap();
                        let exp = tok.encode_fast(prompt.as_str(), add_spec).unwrap();
                        assert_eq!(got.as_slice(), exp.get_ids());
                    }
                });
            }
        });
    }
}
