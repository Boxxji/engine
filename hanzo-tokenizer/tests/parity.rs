// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: MIT

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use hanzo_tokenizer::{PromptCacheStats, PromptTokenCache, Tokenizer, TokenizerError};
use serde_json::json;

fn cached_hf_tokenizer(repo_dir: &str) -> Option<PathBuf> {
    let base = PathBuf::from("/home/z/.cache/huggingface/hub")
        .join(repo_dir)
        .join("snapshots");
    let entries = std::fs::read_dir(base).ok()?;
    for entry in entries.flatten() {
        let cand = entry.path().join("tokenizer.json");
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

fn synthetic_byte_level_config(
    normalizer: serde_json::Value,
    pre_tokenizer: serde_json::Value,
    post_processor: serde_json::Value,
    added_tokens: serde_json::Value,
    ignore_merges: bool,
    array_merges: bool,
) -> Vec<u8> {
    let mut vocab = BTreeMap::new();
    let mut id = 0u32;
    let base_chars: Vec<String> = {
        let mut bs: Vec<u8> = (b'!'..=b'~')
            .chain(0xA1..=0xAC)
            .chain(0xAE..=0xFF)
            .collect();
        let mut out = Vec::new();
        for b in 0u16..=255 {
            let b = b as u8;
            if let Some(pos) = bs.iter().position(|&x| x == b) {
                let _ = pos;
                out.push((b as char).to_string());
            } else {
                let extra_idx = (0u8..b)
                    .filter(|&x| {
                        !((b'!'..=b'~').contains(&x)
                            || (0xA1..=0xAC).contains(&x)
                            || (0xAE..=0xFF).contains(&x))
                    })
                    .count();
                let cp = 256u32 + extra_idx as u32;
                out.push(char::from_u32(cp).unwrap().to_string());
            }
        }
        let _ = &mut bs;
        out
    };
    for s in base_chars {
        vocab.insert(s, id);
        id += 1;
    }

    let extra_tokens = [
        "he", "ll", "lo", "hel", "llo", "hello", "wo", "rl", "wor", "ld", "world", "Ġw", "Ġwo",
        "Ġwor", "Ġworld", "foo", "fo", "bar", "ba", "<unk>", "<bos>", "<eos>", "<pad>", "<spec>",
        "<norm>",
    ];
    for t in extra_tokens {
        vocab.entry(t.to_string()).or_insert_with(|| {
            let cur = id;
            id += 1;
            cur
        });
    }

    let raw_merges = [
        ("h", "e"),
        ("l", "l"),
        ("ll", "o"),
        ("he", "llo"),
        ("w", "o"),
        ("r", "l"),
        ("wo", "r"),
        ("wor", "ld"),
        ("Ġ", "w"),
        ("Ġw", "o"),
        ("Ġwo", "r"),
        ("f", "o"),
        ("fo", "o"),
        ("b", "a"),
        ("ba", "r"),
    ];
    let merges: serde_json::Value = if array_merges {
        serde_json::Value::Array(raw_merges.iter().map(|(a, b)| json!([a, b])).collect())
    } else {
        serde_json::Value::Array(
            raw_merges
                .iter()
                .map(|(a, b)| json!(format!("{a} {b}")))
                .collect(),
        )
    };

    serde_json::to_vec(&json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": normalizer,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": post_processor,
        "decoder": null,
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": "<unk>",
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": ignore_merges,
            "vocab": vocab,
            "merges": merges
        }
    }))
    .unwrap()
}

fn synthetic_char_bpe_config(
    normalizer: serde_json::Value,
    pre_tokenizer: serde_json::Value,
    byte_fallback: bool,
    fuse_unk: bool,
    continuing_subword_prefix: Option<&str>,
    end_of_word_suffix: Option<&str>,
    unk_token: Option<&str>,
) -> Vec<u8> {
    let mut vocab = BTreeMap::new();
    let mut id = 0u32;
    let mut insert = |s: &str| {
        if !vocab.contains_key(s) {
            vocab.insert(s.to_string(), id);
            id += 1;
        }
    };

    insert("<unk>");
    if byte_fallback {
        for b in 0u16..=255 {
            insert(&format!("<{:#04X}>", b as u8));
        }
    }
    for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !.,:-_".chars() {
        let s = ch.to_string();
        insert(&s);
        if let Some(p) = continuing_subword_prefix {
            insert(&format!("{p}{s}"));
        }
        if let Some(suf) = end_of_word_suffix {
            insert(&format!("{s}{suf}"));
        }
        if let (Some(p), Some(suf)) = (continuing_subword_prefix, end_of_word_suffix) {
            insert(&format!("{p}{s}{suf}"));
        }
    }
    for extra in [
        "ab", "bc", "abc", "12", "23", "123", "##bc", "bc</w>", "abc</w>", "##bc</w>", "123</w>",
        "a</w>", "b</w>", "c</w>", "1</w>", "2</w>", "3</w>", "##c</w>",
    ] {
        insert(extra);
    }

    let p = continuing_subword_prefix.unwrap_or("");
    let s = end_of_word_suffix.unwrap_or("");
    let merges = vec![
        format!("a {p}b"),
        format!("ab {p}c{s}"),
        format!("{p}b {p}c{s}"),
        format!("1 {p}2"),
        format!("12 {p}3{s}"),
    ];

    serde_json::to_vec(&json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": normalizer,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": unk_token,
            "continuing_subword_prefix": continuing_subword_prefix,
            "end_of_word_suffix": end_of_word_suffix,
            "fuse_unk": fuse_unk,
            "byte_fallback": byte_fallback,
            "ignore_merges": false,
            "vocab": vocab,
            "merges": merges
        }
    }))
    .unwrap()
}

fn assert_parity_bytes(json_bytes: &[u8], samples: &[&str]) {
    let hf = tokenizers::Tokenizer::from_bytes(json_bytes).expect("hf load");
    let hz = Tokenizer::from_bytes(json_bytes).expect("hz load");

    for &sample in samples {
        for &add_special in &[false, true] {
            let expected = hf
                .encode(sample, add_special)
                .expect("hf encode")
                .get_ids()
                .to_vec();
            let actual = hz
                .encode_with_special_tokens(sample, add_special)
                .expect("hz encode");
            assert_eq!(
                actual, expected,
                "mismatch for add_special={add_special}, sample={sample:?}"
            );
        }
        let default_expected = hf.encode(sample, true).unwrap().get_ids().to_vec();
        let default_actual = hz.encode(sample).unwrap();
        assert_eq!(default_actual, default_expected);
    }
}

#[test]
fn synthetic_normalizers_parity() {
    let normalizers = [
        json!(null),
        json!({"type": "NFC"}),
        json!({"type": "NFKC"}),
        json!({"type": "NFD"}),
        json!({"type": "NFKD"}),
        json!({"type": "Lowercase"}),
        json!({"type": "Prepend", "prepend": ">>"}),
        json!({"type": "Strip", "strip_left": true, "strip_right": true}),
        json!({"type": "Replace", "pattern": {"String": "foo"}, "content": "bar"}),
        json!({"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "}),
        json!({"type": "Sequence", "normalizers": [
            {"type": "NFC"},
            {"type": "Lowercase"},
            {"type": "Strip", "strip_left": true, "strip_right": true},
            {"type": "Replace", "pattern": {"String": "  "}, "content": " "}
        ]}),
    ];

    let pre = json!({
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": false,
        "use_regex": false
    });
    let samples = [
        "",
        "hello world",
        "  Foo  WORLD  ",
        "cafe\u{0301}  CAFÉ",
        "ﬁ ﬂ ① ② ²",
        "foo   bar   hello",
        "   ",
    ];

    for (idx, norm) in normalizers.into_iter().enumerate() {
        let cfg = synthetic_byte_level_config(
            norm,
            pre.clone(),
            json!(null),
            json!([]),
            idx % 2 == 0,
            idx % 2 == 1,
        );
        assert_parity_bytes(&cfg, &samples);
    }
}

#[test]
fn synthetic_pre_tokenizers_and_ignore_merges_parity() {
    let pre_tokenizers = [
        json!(null),
        json!({"type": "Digits", "individual_digits": true}),
        json!({"type": "Digits", "individual_digits": false}),
        json!({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true}),
        json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true}),
        json!({"type": "Split", "pattern": {"String": " "}, "behavior": "Isolated", "invert": false}),
        json!({"type": "Split", "pattern": {"String": " "}, "behavior": "Removed", "invert": false}),
        json!({"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": false}),
        json!({"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithNext", "invert": false}),
        json!({"type": "Split", "pattern": {"Regex": "[a-z]+"}, "behavior": "Isolated", "invert": true}),
        json!({"type": "Split", "pattern": {"Regex": "\\w+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}),
        json!({"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
        ]}),
    ];

    let samples = [
        "",
        "hello world",
        " hello   world!  \n\nI'm   12345  ",
        "foobar hello world",
        "a b c   d\te\nf",
        "123abc456def",
        "   ",
    ];

    for (i, pre) in pre_tokenizers.into_iter().enumerate() {
        let is_byte_level = serde_json::to_string(&pre).unwrap().contains("ByteLevel");
        let cfg = if is_byte_level {
            synthetic_byte_level_config(json!(null), pre, json!(null), json!([]), i % 2 == 0, true)
        } else {
            synthetic_char_bpe_config(json!(null), pre, false, false, None, None, Some("<unk>"))
        };
        assert_parity_bytes(&cfg, &samples);
    }

    // Verify ignore_merges forces direct whole-word lookup without merging "Ġwor" + "ld" -> wait
    // Note in synthetic vocab "Ġworld" IS in vocab, while merge table stops at "Ġwor" without merging "Ġwor ld"!
    let pre_seq = json!({
        "type": "Sequence",
        "pretokenizers": [
            {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithNext", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
        ]
    });
    let cfg_ignore = synthetic_byte_level_config(
        json!(null),
        pre_seq.clone(),
        json!(null),
        json!([]),
        true,
        true,
    );
    let cfg_no_ignore =
        synthetic_byte_level_config(json!(null), pre_seq, json!(null), json!([]), false, true);
    let hz_ign = Tokenizer::from_bytes(&cfg_ignore).unwrap();
    let hz_no = Tokenizer::from_bytes(&cfg_no_ignore).unwrap();
    let t_ign = hz_ign.encode("hello world").unwrap();
    let t_no = hz_no.encode("hello world").unwrap();
    assert_ne!(t_ign, t_no);
    assert_parity_bytes(&cfg_ignore, &["hello world", "foo bar"]);
    assert_parity_bytes(&cfg_no_ignore, &["hello world", "foo bar"]);
}

#[test]
fn synthetic_char_bpe_byte_fallback_fuse_unk_prefix_suffix_parity() {
    let cases = [
        (true, false, None, None, Some("<unk>")),
        (false, true, None, None, Some("<unk>")),
        (false, false, Some("##"), None, Some("<unk>")),
        (false, false, None, Some("</w>"), Some("<unk>")),
        (false, false, Some("##"), Some("</w>"), Some("<unk>")),
    ];
    let pre = json!({"type": "Split", "pattern": {"String": " "}, "behavior": "Removed", "invert": false});
    let samples = [
        "abc 123",
        "abc ??? 123",
        "hello 世界 café 🚀",
        "abc def ghi",
    ];
    for (bf, fu, cp, eow, unk) in cases {
        let cfg = synthetic_char_bpe_config(json!(null), pre.clone(), bf, fu, cp, eow, unk);
        assert_parity_bytes(&cfg, &samples);
    }
}

#[test]
fn synthetic_added_tokens_and_template_post_processor_parity() {
    let added = json!([
        {"id": 276, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 277, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 279, "content": "<spec>", "single_word": true, "lstrip": true, "rstrip": true, "normalized": false, "special": false},
        {"id": 281, "content": "<NORM>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": false},
        {"id": 278, "content": "<pad>", "single_word": true, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
    ]);
    let post = json!({
        "type": "Sequence",
        "processors": [
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
            {
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "<eos>", "type_id": 0}}
                ],
                "pair": [
                    {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "<eos>", "type_id": 0}},
                    {"Sequence": {"id": "B", "type_id": 1}},
                    {"SpecialToken": {"id": "<eos>", "type_id": 1}}
                ],
                "special_tokens": {
                    "<bos>": {"id": "<bos>", "ids": [276], "tokens": ["<bos>"]},
                    "<eos>": {"id": "<eos>", "ids": [277], "tokens": ["<eos>"]}
                }
            }
        ]
    });
    let pre = json!({
        "type": "Sequence",
        "pretokenizers": [
            {"type": "Split", "pattern": {"String": " "}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
        ]
    });
    let cfg =
        synthetic_byte_level_config(json!({"type": "Lowercase"}), pre, post, added, false, true);
    let samples = [
        "",
        "hello world",
        "<bos>hello<eos>",
        "hello   <spec>   world",
        "foo<spec>bar",
        "foo <spec>bar",
        "<norm> <NORM> hello <pad> foo<pad>bar",
        "hello  <pad>  <spec>  world",
    ];
    assert_parity_bytes(&cfg, &samples);
}

#[test]
fn unsupported_features_return_explicit_errors() {
    let base_pre = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false});

    // 1. Unknown normalizer
    let bad_norm = synthetic_byte_level_config(
        json!({"type": "BertNormalizer"}),
        base_pre.clone(),
        json!(null),
        json!([]),
        false,
        true,
    );
    assert!(matches!(
        Tokenizer::from_bytes(&bad_norm),
        Err(TokenizerError::Unsupported(_))
    ));

    // 2. Arbitrary lookaround in Split
    let bad_split = synthetic_byte_level_config(
        json!(null),
        json!({"type": "Split", "pattern": {"Regex": "(?<=a)b"}, "behavior": "Isolated", "invert": false}),
        json!(null),
        json!([]),
        false,
        true,
    );
    assert!(matches!(
        Tokenizer::from_bytes(&bad_split),
        Err(TokenizerError::Unsupported(_))
    ));

    // 3. Unsupported TemplateProcessing with Sequence B in single
    let bad_pp = synthetic_byte_level_config(
        json!(null),
        base_pre.clone(),
        json!({
            "type": "TemplateProcessing",
            "single": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
            "special_tokens": {}
        }),
        json!([]),
        false,
        true,
    );
    assert!(matches!(
        Tokenizer::from_bytes(&bad_pp),
        Err(TokenizerError::Unsupported(_))
    ));

    // 4. Non-null truncation / padding / non-1.0 version / WordPiece model
    let mut doc: serde_json::Value = serde_json::from_slice(&synthetic_byte_level_config(
        json!(null),
        base_pre,
        json!(null),
        json!([]),
        false,
        true,
    ))
    .unwrap();

    let mut doc_trunc = doc.clone();
    doc_trunc["truncation"] = json!({"max_length": 128});
    assert!(matches!(
        Tokenizer::from_bytes(&serde_json::to_vec(&doc_trunc).unwrap()),
        Err(TokenizerError::Unsupported(_))
    ));

    let mut doc_pad = doc.clone();
    doc_pad["padding"] = json!({"strategy": "BatchLongest"});
    assert!(matches!(
        Tokenizer::from_bytes(&serde_json::to_vec(&doc_pad).unwrap()),
        Err(TokenizerError::Unsupported(_))
    ));

    let mut doc_ver = doc.clone();
    doc_ver["version"] = json!("2.0");
    assert!(matches!(
        Tokenizer::from_bytes(&serde_json::to_vec(&doc_ver).unwrap()),
        Err(TokenizerError::Unsupported(_))
    ));

    doc["model"]["type"] = json!("WordPiece");
    assert!(matches!(
        Tokenizer::from_bytes(&serde_json::to_vec(&doc).unwrap()),
        Err(TokenizerError::Unsupported(_))
    ));
}

#[test]
fn cached_hf_models_full_parity() {
    let models = [
        "models--Qwen--Qwen3-8B",
        "models--hoangquan456--Kimi-K2.5",
        "models--openai--gpt-oss-120b",
        "models--zai-org--GLM-4.7",
        "models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "models--deepseek-ai--DeepSeek-V3.2",
    ];

    let corpus = [
        "",
        " ",
        "   ",
        "\n\n\n  \t \r\n",
        "Hello, world!",
        "I'm don't we're they've I'll he'd IT'S WE'RE",
        "foo1234567890bar 1 12 123 1234 999999999",
        "http://example.com/a/b/c?x=1&y=2#frag\n/usr/local/bin/cargo\n// comment /// doc",
        "fn main() -> Result<(), Box<dyn std::error::Error>> {\n    let xs = vec![1, 2, 3];\n    println!(\"{xs:?}\");\n    Ok(())\n}\n",
        "{\"role\":\"assistant\",\"content\":\"hello\\nworld\",\"ok\":true}\n",
        "Café naïve résumé — e\u{0301} a\u{0300} o\u{0302} ﬁ ﬂ ①",
        "你好世界！こんにちは世界。안녕하세요! 123日本語456中文。",
        "مرحبا بالعالم שלום עולם नमस्ते दुनिया ไทย ภาษาไทย",
        "Emoji ZWJ: 👨‍👩‍👧‍👦 👍🏽 🏳️‍🌈 © ™ ∞ ≠ ≈ ∑ √",
        "Mixed spaces:   foo   \n   bar   baz\t\t\r\nqux   ",
        "<|endoftext|>user\nHello!<think>step 1</think>Done.",
    ];

    let large_prompt = {
        let unit = "The quick brown fox jumps over 123 lazy dogs! Café 世界 🚀 // path/to/file\n";
        let mut s = String::with_capacity(unit.len() * 400);
        for _ in 0..400 {
            s.push_str(unit);
        }
        s
    };

    let mut tested_count = 0usize;
    for repo in models {
        let Some(path) = cached_hf_tokenizer(repo) else {
            continue;
        };
        tested_count += 1;
        let bytes = std::fs::read(&path).expect("read cached tokenizer.json");
        let hf = tokenizers::Tokenizer::from_bytes(&bytes).expect("hf load");
        let hz = Tokenizer::from_bytes(&bytes).expect("hz load");

        for &sample in &corpus {
            for &add_special in &[false, true] {
                let expected = hf.encode(sample, add_special).unwrap().get_ids().to_vec();
                let actual = hz.encode_with_special_tokens(sample, add_special).unwrap();
                assert_eq!(
                    actual, expected,
                    "parity mismatch in {repo} (add_special={add_special}) on {sample:?}"
                );
            }
        }

        let expected_large = hf
            .encode(large_prompt.as_str(), true)
            .unwrap()
            .get_ids()
            .to_vec();
        let actual_large = hz.encode(&large_prompt).unwrap();
        assert_eq!(
            actual_large, expected_large,
            "large parallel parity mismatch in {repo}"
        );
    }

    assert!(
        tested_count > 0,
        "expected at least one cached HF model under /home/z/.cache/huggingface/hub"
    );
}

#[test]
fn unicode_scalar_parity_against_hf_onig() {
    let models = [
        "models--Qwen--Qwen3-8B",
        "models--hoangquan456--Kimi-K2.5",
        "models--openai--gpt-oss-120b",
        "models--zai-org--GLM-4.7",
        "models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "models--deepseek-ai--DeepSeek-V3.2",
    ];

    // Sample all Unicode scalars in batches separated by lowercase letters and spaces.
    // Using `a{ch}a ` probes both preceding (`U*L+` / `[^\r\n\p{L}\p{N}]?`) and trailing
    // (`U+L*` / `[punct][A-Za-z]+`) transitions across all 6 regex families.
    let all_chars: Vec<char> = (0u32..=0x10_FFFF)
        .filter_map(char::from_u32)
        .filter(|&c| c != '\0')
        .collect();

    for repo in models {
        let Some(path) = cached_hf_tokenizer(repo) else {
            continue;
        };
        let bytes = std::fs::read(&path).expect("read tokenizer.json");
        let hf = tokenizers::Tokenizer::from_bytes(&bytes).expect("hf load");
        let hz = Tokenizer::from_bytes(&bytes).expect("hz load");

        let check_slice = |chars: &[char]| -> bool {
            let mut s = String::with_capacity(chars.len() * 8);
            for &ch in chars {
                s.push('a');
                s.push(ch);
                s.push('a');
                s.push(' ');
            }
            let expected = hf.encode(s.as_str(), false).unwrap().get_ids().to_vec();
            let actual = hz.encode_with_special_tokens(&s, false).unwrap();
            expected == actual
        };

        for chunk in all_chars.chunks(256) {
            if !check_slice(chunk) {
                let mut lo = 0usize;
                let mut hi = chunk.len();
                while hi - lo > 1 {
                    let mid = (lo + hi) / 2;
                    if check_slice(&chunk[lo..mid]) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let bad = chunk[lo];
                let probes = [
                    format!("{bad}"),
                    format!("a{bad}a "),
                    format!("a{bad}b"),
                    format!(" {bad} "),
                    format!("\n{bad}x"),
                ];
                for p in probes {
                    let exp = hf.encode(p.as_str(), false).unwrap().get_ids().to_vec();
                    let act = hz.encode_with_special_tokens(&p, false).unwrap();
                    assert_eq!(
                        act, exp,
                        "Unicode parity mismatch in {repo} for U+{:04X} ({bad:?}) on {p:?}",
                        bad as u32
                    );
                }
                panic!(
                    "Unicode parity batch mismatch in {repo} around U+{:04X} ({bad:?})",
                    bad as u32
                );
            }
        }
    }
}

#[test]
fn file_loading_and_invalid_config_errors() {
    let missing = Tokenizer::from_file("/nonexistent/path/to/tokenizer.json");
    assert!(matches!(missing, Err(TokenizerError::Io(_))));

    let bad_json = Tokenizer::from_bytes(b"{not valid json");
    assert!(matches!(bad_json, Err(TokenizerError::Json(_))));

    let bad_merge = json!({
        "version": "1.0",
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "model": {
            "type": "BPE",
            "continuing_subword_prefix": "##",
            "vocab": {"a": 0, "b": 1, "ab": 2},
            "merges": ["a b"]
        }
    });
    assert!(matches!(
        Tokenizer::from_bytes(&serde_json::to_vec(&bad_merge).unwrap()),
        Err(TokenizerError::InvalidConfig(_))
    ));

    if let Some(path) = cached_hf_tokenizer("models--Qwen--Qwen3-8B") {
        let tok = Tokenizer::from_file(path).unwrap();
        assert!(!tok.encode("hello").unwrap().is_empty());
    }
}

#[test]
fn prompt_cache_hits_boundary_safety_and_auto_bypass() {
    // 1. Test TemplateProcessing (BOS + EOS) + ByteLevel checkpoint parity
    let pre = json!({
        "type": "Sequence",
        "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
        ]
    });
    let added = json!([
        {"id": 276, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 277, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
    ]);
    let post = json!({
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "<bos>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "<eos>", "type_id": 0}}
        ],
        "special_tokens": {
            "<bos>": {"id": "<bos>", "ids": [276], "tokens": ["<bos>"]},
            "<eos>": {"id": "<eos>", "ids": [277], "tokens": ["<eos>"]}
        }
    });
    let cfg = synthetic_byte_level_config(json!({"type": "NFC"}), pre, post, added, false, true);
    let hz = Tokenizer::from_bytes(&cfg).unwrap();
    let mut cache = PromptTokenCache::new(4);

    // Multi-turn incremental growth with non-ASCII NFD->NFC before `\n`!
    let t1 = "Cafe\u{0301} line 1\n";
    let t2 = "Cafe\u{0301} line 1\nhello world\n";
    let t3 = "Cafe\u{0301} line 1\nhello world\n<bos>tail\n";
    let t4_slash = "Cafe\u{0301} line 1\nhello world\n/slash/path";
    let t5_quote = "Cafe\u{0301} line 1\nhello world\n's contraction";

    for &prompt in &[t1, t2, t3, t4_slash, t5_quote] {
        let cached_out = hz.encode_with_cache(prompt, true, &mut cache).unwrap();
        let uncached_out = hz.encode_with_special_tokens(prompt, true).unwrap();
        assert_eq!(cached_out, uncached_out, "prompt={prompt:?}");
    }

    let stats = cache.stats();
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.prefix_hits, 4);

    // Exact hit
    assert_eq!(
        hz.encode_with_cache(t2, true, &mut cache).unwrap(),
        hz.encode_with_special_tokens(t2, true).unwrap()
    );
    assert_eq!(cache.stats().exact_hits, 1);

    // Zero-capacity cache always bypasses
    let mut zero_cache = PromptTokenCache::new(0);
    assert_eq!(
        hz.encode_with_cache(t1, true, &mut zero_cache).unwrap(),
        hz.encode_with_special_tokens(t1, true).unwrap()
    );
    assert_eq!(
        zero_cache.stats(),
        PromptCacheStats {
            exact_hits: 0,
            prefix_hits: 0,
            misses: 0,
            bypassed: 1,
        }
    );

    // 2. Test cached HF models with PromptTokenCache multi-turn + auto-bypass
    for repo in [
        "models--Qwen--Qwen3-8B",
        "models--openai--gpt-oss-120b",
        "models--deepseek-ai--DeepSeek-V3.2",
    ] {
        let Some(path) = cached_hf_tokenizer(repo) else {
            continue;
        };
        let model_tok = Tokenizer::from_file(path).unwrap();
        let mut mcache = PromptTokenCache::new(8);

        let mut acc = String::from("<|im_start|>system\nYou are helpful!<|im_end|>\n");
        let _ = model_tok
            .encode_with_cache(&acc, true, &mut mcache)
            .unwrap();
        for turn in 0..5 {
            acc.push_str(&format!("<|im_start|>user\nTurn {turn} café!\n"));
            let got = model_tok
                .encode_with_cache(&acc, true, &mut mcache)
                .unwrap();
            let exp = model_tok.encode_with_special_tokens(&acc, true).unwrap();
            assert_eq!(got, exp, "incremental parity failed on {repo} turn {turn}");
        }
        assert!(mcache.stats().prefix_hits >= 4);

        // Online self-bypass check on 15 completely unrelated single-line misses:
        // 2 consecutive misses -> 8 bypassed -> 1 recovery probe (turns out miss) -> bypassed...
        let mut uc = PromptTokenCache::new(4);
        for i in 0..14 {
            let p = format!("unrelated prompt number {i} without prefix overlap");
            let got = model_tok.encode_with_cache(&p, true, &mut uc).unwrap();
            let exp = model_tok.encode(&p).unwrap();
            assert_eq!(got, exp);
        }
        let ustats = uc.stats();
        assert!(ustats.bypassed >= 8, "expected auto-bypass, got {ustats:?}");
        assert!(
            ustats.misses <= 4,
            "expected bounded misses, got {ustats:?}"
        );

        // Recovery after cooldown when incremental prefix resumes:
        // First flush remaining 5 cooldown calls, insert `p_base` on the recovery probe,
        // flush the subsequent 8 cooldown calls, then verify the next probe hits `p_base`.
        for _ in 0..5 {
            let _ = model_tok
                .encode_with_cache("drain cooldown", true, &mut uc)
                .unwrap();
        }
        let p_base = "system header line\n";
        let _ = model_tok.encode_with_cache(p_base, true, &mut uc).unwrap();
        for _ in 0..8 {
            let _ = model_tok
                .encode_with_cache("drain cooldown", true, &mut uc)
                .unwrap();
        }
        let got_rec = model_tok
            .encode_with_cache("system header line\nnext line", true, &mut uc)
            .unwrap();
        assert_eq!(
            got_rec,
            model_tok.encode("system header line\nnext line").unwrap()
        );
        assert!(uc.stats().prefix_hits > 0);

        // Once recovered, consecutive hits stay active with zero bypass increment
        let bypassed_before = uc.stats().bypassed;
        let got_exact = model_tok
            .encode_with_cache("system header line\nnext line", true, &mut uc)
            .unwrap();
        assert_eq!(got_exact, got_rec);
        assert_eq!(uc.stats().exact_hits, 1);
        assert_eq!(uc.stats().bypassed, bypassed_before);
    }
}

#[test]
fn concurrent_16_thread_encoding_parity() {
    let pre = json!({
        "type": "Sequence",
        "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
        ]
    });
    let tok = if let Some(path) = cached_hf_tokenizer("models--Qwen--Qwen3-8B") {
        Tokenizer::from_file(path).unwrap()
    } else {
        let cfg = synthetic_byte_level_config(
            json!({"type": "NFC"}),
            pre,
            json!(null),
            json!([]),
            false,
            true,
        );
        Tokenizer::from_bytes(&cfg).unwrap()
    };

    let samples: Vec<String> = (0..16)
        .map(|i| {
            format!(
                "Thread {i}: hello world! Café 你好 世界 🚀 // path/item_{i} 123456\n{}\n",
                "repeated_segment ".repeat(20)
            )
        })
        .collect();
    let expected: Vec<Vec<u32>> = samples.iter().map(|s| tok.encode(s).unwrap()).collect();

    let shared = Arc::new(tok);
    let handles: Vec<_> = (0..16)
        .map(|tid| {
            let t = Arc::clone(&shared);
            let s = samples[tid].clone();
            let exp = expected[tid].clone();
            thread::spawn(move || {
                for _ in 0..25 {
                    let got = t.encode(&s).unwrap();
                    assert_eq!(got, exp);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("worker thread panicked");
    }
}
