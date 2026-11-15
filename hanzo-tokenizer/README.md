# `hanzo-tokenizer`

Copyright 2026 Hanzo AI, Inc. — Licensed under the [MIT License](../LICENSE).

`hanzo-tokenizer` is a pure-Rust, clean-room BPE tokenizer supporting standard Hugging Face `tokenizer.json` configurations with deterministic linear-time pre-tokenization scanners, open-addressed BPE merge tables, and boundary-safe incremental prompt prefix caching.

## Public API

- `Tokenizer`: `from_bytes`, `from_file`, `encode`, `encode_with_special_tokens`, `encode_with_cache` (`Send + Sync`, `Clone`).
- `PromptTokenCache` / `PromptCacheStats`: bounded LRU cache supporting exact hits, incremental `\n` prefix checkpoint reuse, and automatic self-bypass cooldowns on non-incremental workloads.
- `TokenizerError`: explicit error enum (`Io`, `Json`, `InvalidConfig`, `Unsupported`, `MissingVocabToken`, `Regex`).

## Supported `tokenizer.json` Features

- **Added tokens**: raw (`normalized: false`) and normalized (`normalized: true`) Aho-Corasick `LeftmostLongest` matching with `lstrip`, `rstrip`, and `single_word` support.
- **Normalizers**: `null`, `NFC`, `NFKC`, `NFD`, `NFKD`, `Lowercase`, `Prepend`, `Strip`, `Replace` (`String` and lookaround-free `Regex`), `Sequence` (with pure-ASCII no-allocation fast paths).
- **Pre-tokenizers**: `null`, `ByteLevel`, `Digits`, `Split` (`Isolated`, `Removed`, `MergedWithPrevious`, `MergedWithNext`, `invert`, lookaround-free `Regex`, linear-time `|\s+(?!\S)|\s+` handling, and dedicated deterministic $O(n)$ Unicode scanners for CL100K/Qwen/GLM, O200K/GPT-OSS/Nemotron/Kimi, DeepSeek-V3 digits/CJK/main, and GPT-2 patterns), and `Sequence`.
- **Model**: `BPE` (string and 2-element array merges, `unk_token`, `fuse_unk`, `byte_fallback`, `ignore_merges`, `continuing_subword_prefix`, `end_of_word_suffix`).
- **Post-processor**: `null`, `ByteLevel`, `TemplateProcessing` (single-sequence `A` templates with prefix/suffix `SpecialToken` entries), and `Sequence`.

Unsupported configurations (such as `truncation`, `padding`, non-`"1.0"` versions, non-`BPE` models, unknown normalizer/pre-tokenizer/post-processor types, arbitrary regex lookarounds, or pair/`Sequence("B")` single templates) return `Err(TokenizerError::Unsupported(...))` rather than silently mis-tokenizing.

## Verification

```bash
cargo test -p hanzo-tokenizer
cargo clippy -p hanzo-tokenizer --tests -- -D warnings
cargo fmt -p hanzo-tokenizer -- --check
```
