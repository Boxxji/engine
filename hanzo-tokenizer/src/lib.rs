// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: MIT

//! Clean-room fast Hugging Face `tokenizer.json` BPE tokenizer with SIMD/SWAR
//! pre-tokenization scanners, stacked BPE fast paths, and boundary-safe
//! incremental prefix caching.

pub mod bpe;
pub mod normalizer;
pub mod pre_tokenizer;
pub mod prompt_cache;
mod unicode_lut;

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use rayon::prelude::*;
use regex::Regex;
use rustc_hash::FxHashMap;
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use crate::bpe::BpeModel;
use crate::normalizer::Normalizer;
use crate::pre_tokenizer::PreTokenizer;
pub use crate::prompt_cache::{PromptCacheStats, PromptTokenCache};

pub(crate) type EncodedWithCheckpoints = (Vec<u32>, Vec<(usize, usize)>);

static NEXT_TOKENIZER_ID: AtomicU64 = AtomicU64::new(1);

const PARALLEL_BYTE_THRESHOLD: usize = 16 * 1024;
const PARALLEL_MIN_PIECES: usize = 512;
const PARALLEL_CHUNK_PIECES: usize = 256;

static ENDS_WITH_WORD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\w$").expect("valid word suffix regex"));
static STARTS_WITH_WORD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\w").expect("valid word prefix regex"));

/// Errors returned while loading or executing [`Tokenizer`].
#[derive(Debug, Error)]
pub enum TokenizerError {
    /// Filesystem read failure while loading a tokenizer file.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Invalid or malformed JSON in `tokenizer.json`.
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    /// Structurally invalid or inconsistent tokenizer configuration.
    #[error("invalid tokenizer config: {0}")]
    InvalidConfig(String),
    /// Tokenizer feature or configuration variant intentionally rejected rather than mis-tokenized.
    #[error("unsupported tokenizer feature: {0}")]
    Unsupported(String),
    /// Vocabulary lookup failed for a required special, unknown, or merge token.
    #[error("missing token in vocabulary: {0}")]
    MissingVocabToken(String),
    /// Invalid regex pattern encountered in tokenizer configuration.
    #[error("regex compilation error: {0}")]
    Regex(#[from] regex::Error),
}

/// Fast Hugging Face `tokenizer.json` BPE tokenizer.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    inner: Arc<TokenizerInner>,
}

#[derive(Debug)]
struct TokenizerInner {
    id: u64,
    added_vocab: AddedVocabulary,
    normalizer: Normalizer,
    pre_tokenizer: PreTokenizer,
    bpe: BpeModel,
    post_processor: PostProcessor,
    fast_byte_level: bool,
    supports_checkpoints: bool,
}

impl Tokenizer {
    /// Parses a Hugging Face `tokenizer.json` document from raw bytes.
    pub fn from_bytes(json_bytes: &[u8]) -> Result<Self, TokenizerError> {
        let root: Value = serde_json::from_slice(json_bytes)?;
        let obj = root.as_object().ok_or_else(|| {
            TokenizerError::InvalidConfig("tokenizer.json root must be a JSON object".to_string())
        })?;

        if let Some(version_val) = obj.get("version") {
            if !version_val.is_null() {
                let ver = version_val.as_str().ok_or_else(|| {
                    TokenizerError::InvalidConfig(
                        "tokenizer `version` must be a string".to_string(),
                    )
                })?;
                if ver != "1.0" {
                    return Err(TokenizerError::Unsupported(format!(
                        "unsupported tokenizer.json version: {ver}"
                    )));
                }
            }
        }

        if let Some(trunc) = obj.get("truncation") {
            if !trunc.is_null() {
                return Err(TokenizerError::Unsupported(
                    "truncation is not supported".to_string(),
                ));
            }
        }

        if let Some(pad) = obj.get("padding") {
            if !pad.is_null() {
                return Err(TokenizerError::Unsupported(
                    "padding is not supported".to_string(),
                ));
            }
        }

        let normalizer = Normalizer::from_value(obj.get("normalizer"))?;
        let pre_tokenizer = PreTokenizer::from_value(obj.get("pre_tokenizer"))?;
        let model_val = obj.get("model").ok_or_else(|| {
            TokenizerError::InvalidConfig("tokenizer.json missing `model` object".to_string())
        })?;
        let bpe = BpeModel::from_value(model_val)?;
        let post_processor = PostProcessor::from_value(obj.get("post_processor"))?;
        let added_vocab = AddedVocabulary::from_value(obj.get("added_tokens"), &normalizer)?;

        let fast_byte_level = pre_tokenizer.is_pure_byte_level_no_prefix();
        let supports_checkpoints = fast_byte_level
            && normalizer.is_boundary_safe_for_newline()
            && pre_tokenizer.is_boundary_safe_for_newline()
            && added_vocab.is_newline_checkpoint_safe();

        let id = NEXT_TOKENIZER_ID.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            inner: Arc::new(TokenizerInner {
                id,
                added_vocab,
                normalizer,
                pre_tokenizer,
                bpe,
                post_processor,
                fast_byte_level,
                supports_checkpoints,
            }),
        })
    }

    /// Reads and parses a Hugging Face `tokenizer.json` file from disk.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Encodes `text` with `add_special_tokens = true`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.encode_with_special_tokens(text, true)
    }

    /// Encodes `text`, optionally applying template post-processor special tokens.
    pub fn encode_with_special_tokens(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, TokenizerError> {
        let mut out = Vec::with_capacity(text.len() / 3 + 8);
        if add_special_tokens {
            out.extend_from_slice(&self.inner.post_processor.prefix_ids);
        }

        if !text.is_empty() {
            if !self.inner.added_vocab.has_non_normalized() {
                self.encode_raw_slice(text, &mut out)?;
            } else {
                let stage1 = self.inner.added_vocab.split_non_normalized(text);
                for seg in stage1 {
                    match seg {
                        AddedSplit::Token(id) => out.push(id),
                        AddedSplit::Raw(start, end) => {
                            self.encode_raw_slice(&text[start..end], &mut out)?;
                        }
                    }
                }
            }
        }

        if add_special_tokens {
            out.extend_from_slice(&self.inner.post_processor.suffix_ids);
        }
        Ok(out)
    }

    /// Encodes `text` using `cache` for exact hits, boundary-safe incremental `\n`
    /// prefix reuse, and automatic bypass on non-incremental workloads.
    pub fn encode_with_cache(
        &self,
        text: &str,
        add_special_tokens: bool,
        cache: &mut PromptTokenCache,
    ) -> Result<Vec<u32>, TokenizerError> {
        cache.encode_with(self, text, add_special_tokens)
    }

    #[inline]
    pub(crate) fn id(&self) -> u64 {
        self.inner.id
    }

    #[inline]
    pub(crate) fn supports_newline_checkpoints(&self) -> bool {
        self.inner.supports_checkpoints
    }

    pub(crate) fn encode_with_checkpoints(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<EncodedWithCheckpoints, TokenizerError> {
        self.encode_body_with_checkpoints(text, add_special_tokens, add_special_tokens, 0, 0)
    }

    pub(crate) fn encode_incremental_tail(
        &self,
        tail: &str,
        add_special_tokens: bool,
        base_byte_offset: usize,
        base_token_offset: usize,
    ) -> Result<EncodedWithCheckpoints, TokenizerError> {
        self.encode_body_with_checkpoints(
            tail,
            false,
            add_special_tokens,
            base_byte_offset,
            base_token_offset,
        )
    }

    fn encode_body_with_checkpoints(
        &self,
        input: &str,
        add_bos: bool,
        add_eos: bool,
        base_byte_offset: usize,
        base_token_offset: usize,
    ) -> Result<EncodedWithCheckpoints, TokenizerError> {
        let mut out = Vec::with_capacity(input.len() / 3 + 8);
        let mut checkpoints = Vec::new();
        if add_bos {
            out.extend_from_slice(&self.inner.post_processor.prefix_ids);
        }

        if !input.is_empty() {
            if !self.inner.added_vocab.has_non_normalized() {
                self.encode_raw_slice_with_checkpoints(
                    input,
                    0,
                    input,
                    base_byte_offset,
                    base_token_offset,
                    &mut out,
                    &mut checkpoints,
                )?;
            } else {
                let stage1 = self.inner.added_vocab.split_non_normalized(input);
                for seg in stage1 {
                    match seg {
                        AddedSplit::Token(id) => out.push(id),
                        AddedSplit::Raw(start, end) => {
                            self.encode_raw_slice_with_checkpoints(
                                input,
                                start,
                                &input[start..end],
                                base_byte_offset,
                                base_token_offset,
                                &mut out,
                                &mut checkpoints,
                            )?;
                        }
                    }
                }
            }
        }

        if add_eos {
            out.extend_from_slice(&self.inner.post_processor.suffix_ids);
        }
        Ok((out, checkpoints))
    }

    fn encode_raw_slice(&self, raw: &str, out: &mut Vec<u32>) -> Result<(), TokenizerError> {
        if raw.is_empty() {
            return Ok(());
        }
        let norm = self.inner.normalizer.normalize(raw);
        if norm.is_empty() {
            return Ok(());
        }

        if !self.inner.added_vocab.has_normalized() {
            return self.encode_normalized_segment(&norm, out);
        }

        let stage2 = self.inner.added_vocab.split_normalized(&norm);
        for seg in stage2 {
            match seg {
                AddedSplit::Token(id) => out.push(id),
                AddedSplit::Raw(start, end) => {
                    self.encode_normalized_segment(&norm[start..end], out)?;
                }
            }
        }
        Ok(())
    }

    fn encode_normalized_segment(
        &self,
        segment: &str,
        out: &mut Vec<u32>,
    ) -> Result<(), TokenizerError> {
        if segment.is_empty() {
            return Ok(());
        }

        if self.inner.fast_byte_level {
            if segment.len() >= PARALLEL_BYTE_THRESHOLD && rayon::current_num_threads() > 1 {
                let mut slices = Vec::with_capacity(segment.len() / 4);
                self.inner
                    .pre_tokenizer
                    .split_for_byte_level(segment, |piece, _| slices.push(piece));
                if slices.len() >= PARALLEL_MIN_PIECES {
                    return self.encode_byte_level_slices_parallel(&slices, out);
                }
                for piece in slices {
                    self.inner.bpe.encode_byte_level_piece(piece, out)?;
                }
                return Ok(());
            }

            let mut err: Option<TokenizerError> = None;
            self.inner
                .pre_tokenizer
                .split_for_byte_level(segment, |piece, _| {
                    if err.is_none() {
                        if let Err(e) = self.inner.bpe.encode_byte_level_piece(piece, out) {
                            err = Some(e);
                        }
                    }
                });
            if let Some(e) = err {
                return Err(e);
            }
            return Ok(());
        }

        let pieces = self.inner.pre_tokenizer.pre_tokenize_general(segment);
        if pieces.len() >= PARALLEL_MIN_PIECES && rayon::current_num_threads() > 1 {
            let chunk_results: Vec<Result<Vec<u32>, TokenizerError>> = pieces
                .par_chunks(PARALLEL_CHUNK_PIECES)
                .map(|chunk| {
                    let mut local = Vec::with_capacity(chunk.len() * 2);
                    for (piece, _) in chunk {
                        self.inner.bpe.encode_piece(piece, &mut local)?;
                    }
                    Ok(local)
                })
                .collect();
            for res in chunk_results {
                out.extend_from_slice(&res?);
            }
            return Ok(());
        }

        for (piece, _) in pieces {
            self.inner.bpe.encode_piece(&piece, out)?;
        }
        Ok(())
    }

    fn encode_byte_level_slices_parallel(
        &self,
        slices: &[&str],
        out: &mut Vec<u32>,
    ) -> Result<(), TokenizerError> {
        let chunk_results: Vec<Result<Vec<u32>, TokenizerError>> = slices
            .par_chunks(PARALLEL_CHUNK_PIECES)
            .map(|chunk| {
                let mut local = Vec::with_capacity(chunk.len() * 2);
                for piece in chunk {
                    self.inner.bpe.encode_byte_level_piece(piece, &mut local)?;
                }
                Ok(local)
            })
            .collect();
        for res in chunk_results {
            out.extend_from_slice(&res?);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_raw_slice_with_checkpoints(
        &self,
        full_input: &str,
        raw_base: usize,
        raw_slice: &str,
        base_byte_offset: usize,
        base_token_offset: usize,
        out: &mut Vec<u32>,
        checkpoints: &mut Vec<(usize, usize)>,
    ) -> Result<(), TokenizerError> {
        if raw_slice.is_empty() {
            return Ok(());
        }

        let is_exact_bytes = self.inner.normalizer.is_noop() || raw_slice.is_ascii();
        let norm = self.inner.normalizer.normalize(raw_slice);
        if norm.is_empty() {
            return Ok(());
        }

        let full_bytes = full_input.as_bytes();

        if !self.inner.added_vocab.has_normalized() {
            let nl_map = if is_exact_bytes {
                None
            } else {
                build_newline_offset_map(raw_slice, raw_base, &norm)
            };

            let mut err: Option<TokenizerError> = None;
            self.inner
                .pre_tokenizer
                .split_for_byte_level(&norm, |piece, local_end| {
                    if err.is_some() {
                        return;
                    }
                    if let Err(e) = self.inner.bpe.encode_byte_level_piece(piece, out) {
                        err = Some(e);
                        return;
                    }
                    if piece.as_bytes().last() == Some(&b'\n') {
                        let mapped_raw = if is_exact_bytes {
                            Some(raw_base + local_end)
                        } else if let Some((raw_nls, norm_nls)) = nl_map.as_ref() {
                            norm_nls
                                .binary_search(&local_end)
                                .ok()
                                .map(|idx| raw_nls[idx])
                        } else {
                            None
                        };
                        if let Some(raw_cp) = mapped_raw {
                            if raw_cp > 0
                                && (raw_cp == full_bytes.len()
                                    || prompt_cache::is_safe_checkpoint_follower(
                                        full_bytes[raw_cp],
                                    ))
                            {
                                checkpoints.push((
                                    base_byte_offset + raw_cp,
                                    base_token_offset + out.len(),
                                ));
                            }
                        }
                    }
                });
            if let Some(e) = err {
                return Err(e);
            }
            return Ok(());
        }

        let stage2 = self.inner.added_vocab.split_normalized(&norm);
        for seg in stage2 {
            match seg {
                AddedSplit::Token(id) => out.push(id),
                AddedSplit::Raw(sub_start, sub_end) => {
                    let sub = &norm[sub_start..sub_end];
                    let mut err: Option<TokenizerError> = None;
                    self.inner
                        .pre_tokenizer
                        .split_for_byte_level(sub, |piece, local_end| {
                            if err.is_some() {
                                return;
                            }
                            if let Err(e) = self.inner.bpe.encode_byte_level_piece(piece, out) {
                                err = Some(e);
                                return;
                            }
                            if is_exact_bytes && piece.as_bytes().last() == Some(&b'\n') {
                                let raw_cp = raw_base + sub_start + local_end;
                                if raw_cp > 0
                                    && (raw_cp == full_bytes.len()
                                        || prompt_cache::is_safe_checkpoint_follower(
                                            full_bytes[raw_cp],
                                        ))
                                {
                                    checkpoints.push((
                                        base_byte_offset + raw_cp,
                                        base_token_offset + out.len(),
                                    ));
                                }
                            }
                        });
                    if let Some(e) = err {
                        return Err(e);
                    }
                }
            }
        }
        Ok(())
    }
}

fn build_newline_offset_map(
    raw_slice: &str,
    raw_base: usize,
    norm_slice: &str,
) -> Option<(Vec<usize>, Vec<usize>)> {
    let raw_nls: Vec<usize> = raw_slice
        .bytes()
        .enumerate()
        .filter_map(|(i, b)| (b == b'\n').then_some(raw_base + i + 1))
        .collect();
    if raw_nls.is_empty() {
        return None;
    }
    let norm_nls: Vec<usize> = norm_slice
        .bytes()
        .enumerate()
        .filter_map(|(i, b)| (b == b'\n').then_some(i + 1))
        .collect();
    if raw_nls.len() == norm_nls.len() {
        Some((raw_nls, norm_nls))
    } else {
        None
    }
}

#[derive(Debug)]
struct AddedTokenMatcher {
    ac: Option<AhoCorasick>,
    entries: Vec<AddedTokenEntry>,
}

#[derive(Debug, Clone, Copy)]
struct AddedTokenEntry {
    id: u32,
    lstrip: bool,
    rstrip: bool,
    single_word: bool,
}

#[derive(Debug)]
struct AddedVocabulary {
    non_normalized: AddedTokenMatcher,
    normalized: AddedTokenMatcher,
    newline_checkpoint_safe: bool,
}

#[derive(Deserialize)]
struct AddedTokenJson {
    id: u32,
    content: String,
    #[serde(default)]
    single_word: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    normalized: bool,
    #[serde(default)]
    #[allow(dead_code)]
    special: bool,
}

#[derive(Debug, Clone, Copy)]
enum AddedSplit {
    Raw(usize, usize),
    Token(u32),
}

impl AddedTokenMatcher {
    fn from_pairs(pairs: Vec<(String, AddedTokenEntry)>) -> Result<Self, TokenizerError> {
        if pairs.is_empty() {
            return Ok(Self {
                ac: None,
                entries: Vec::new(),
            });
        }

        let mut dedup: FxHashMap<String, usize> =
            FxHashMap::with_capacity_and_hasher(pairs.len(), Default::default());
        let mut patterns = Vec::with_capacity(pairs.len());
        let mut entries = Vec::with_capacity(pairs.len());

        for (pat, entry) in pairs {
            if pat.is_empty() {
                continue;
            }
            if let Some(&idx) = dedup.get(&pat) {
                entries[idx] = entry;
            } else {
                let idx = patterns.len();
                dedup.insert(pat.clone(), idx);
                patterns.push(pat);
                entries.push(entry);
            }
        }

        if patterns.is_empty() {
            return Ok(Self {
                ac: None,
                entries: Vec::new(),
            });
        }

        let ac = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&patterns)
            .map_err(|e| TokenizerError::InvalidConfig(format!("added_tokens trie error: {e}")))?;

        Ok(Self {
            ac: Some(ac),
            entries,
        })
    }

    fn split(&self, input: &str) -> Vec<AddedSplit> {
        let Some(ac) = self.ac.as_ref() else {
            return if input.is_empty() {
                Vec::new()
            } else {
                vec![AddedSplit::Raw(0, input.len())]
            };
        };

        let mut out = Vec::new();
        let mut start_offset = 0usize;

        for mat in ac.find_iter(input) {
            let mut start = mat.start();
            let mut stop = mat.end();
            if start < start_offset {
                continue;
            }

            let entry = self.entries[mat.pattern().as_usize()];
            if entry.single_word {
                let start_ok = start == 0 || !ends_with_word(&input[..start]);
                let stop_ok = stop == input.len() || !starts_with_word(&input[stop..]);
                if !start_ok || !stop_ok {
                    continue;
                }
            }

            if entry.lstrip {
                start -= trailing_whitespace_len(&input[start_offset..start]);
            }
            if entry.rstrip {
                stop += leading_whitespace_len(&input[stop..]);
            }

            if start > start_offset {
                out.push(AddedSplit::Raw(start_offset, start));
            }
            out.push(AddedSplit::Token(entry.id));
            start_offset = stop;
        }

        if start_offset < input.len() {
            out.push(AddedSplit::Raw(start_offset, input.len()));
        }
        out
    }
}

impl AddedVocabulary {
    fn from_value(value: Option<&Value>, normalizer: &Normalizer) -> Result<Self, TokenizerError> {
        let Some(val) = value else {
            return Ok(Self {
                non_normalized: AddedTokenMatcher {
                    ac: None,
                    entries: Vec::new(),
                },
                normalized: AddedTokenMatcher {
                    ac: None,
                    entries: Vec::new(),
                },
                newline_checkpoint_safe: true,
            });
        };
        if val.is_null() {
            return Self::from_value(None, normalizer);
        }
        let items: Vec<AddedTokenJson> = serde_json::from_value(val.clone())?;

        let mut non_norm_pairs = Vec::new();
        let mut norm_pairs = Vec::new();
        let mut newline_checkpoint_safe = true;

        for item in items {
            if item.content.is_empty() {
                continue;
            }
            if item.lstrip
                || item.rstrip
                || item.single_word
                || item.content.contains('\n')
                || item.content.contains('\r')
            {
                newline_checkpoint_safe = false;
            }

            let entry = AddedTokenEntry {
                id: item.id,
                lstrip: item.lstrip,
                rstrip: item.rstrip,
                single_word: item.single_word,
            };

            if item.normalized {
                let pat = normalizer.normalize(&item.content).into_owned();
                if pat.contains('\n') || pat.contains('\r') {
                    newline_checkpoint_safe = false;
                }
                norm_pairs.push((pat, entry));
            } else {
                non_norm_pairs.push((item.content, entry));
            }
        }

        Ok(Self {
            non_normalized: AddedTokenMatcher::from_pairs(non_norm_pairs)?,
            normalized: AddedTokenMatcher::from_pairs(norm_pairs)?,
            newline_checkpoint_safe,
        })
    }

    #[inline]
    fn has_non_normalized(&self) -> bool {
        self.non_normalized.ac.is_some()
    }

    #[inline]
    fn has_normalized(&self) -> bool {
        self.normalized.ac.is_some()
    }

    #[inline]
    fn is_newline_checkpoint_safe(&self) -> bool {
        self.newline_checkpoint_safe
    }

    fn split_non_normalized(&self, input: &str) -> Vec<AddedSplit> {
        self.non_normalized.split(input)
    }

    fn split_normalized(&self, input: &str) -> Vec<AddedSplit> {
        self.normalized.split(input)
    }
}

#[inline]
fn is_ascii_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[inline]
fn ends_with_word(s: &str) -> bool {
    let Some(&last) = s.as_bytes().last() else {
        return false;
    };
    if last < 128 {
        is_ascii_word_byte(last)
    } else {
        ENDS_WITH_WORD.is_match(s)
    }
}

#[inline]
fn starts_with_word(s: &str) -> bool {
    let Some(&first) = s.as_bytes().first() else {
        return false;
    };
    if first < 128 {
        is_ascii_word_byte(first)
    } else {
        STARTS_WITH_WORD.is_match(s)
    }
}

#[inline]
fn trailing_whitespace_len(s: &str) -> usize {
    let bytes = s.as_bytes();
    let mut idx = bytes.len();
    while idx > 0 {
        let b = bytes[idx - 1];
        if b < 128 {
            if b.is_ascii_whitespace() {
                idx -= 1;
            } else {
                return bytes.len() - idx;
            }
        } else {
            let trimmed_len = s[..idx].trim_end().len();
            return bytes.len() - trimmed_len;
        }
    }
    bytes.len()
}

#[inline]
fn leading_whitespace_len(s: &str) -> usize {
    let bytes = s.as_bytes();
    let mut idx = 0usize;
    while idx < bytes.len() {
        let b = bytes[idx];
        if b < 128 {
            if b.is_ascii_whitespace() {
                idx += 1;
            } else {
                return idx;
            }
        } else {
            let rest = &s[idx..];
            return idx + (rest.len() - rest.trim_start().len());
        }
    }
    bytes.len()
}

#[derive(Debug, Clone, Default)]
struct PostProcessor {
    prefix_ids: Vec<u32>,
    suffix_ids: Vec<u32>,
}

#[derive(Deserialize)]
struct TemplateSpecialTokenJson {
    ids: Vec<u32>,
}

impl PostProcessor {
    fn from_value(value: Option<&Value>) -> Result<Self, TokenizerError> {
        let Some(val) = value else {
            return Ok(Self::default());
        };
        if val.is_null() {
            return Ok(Self::default());
        }
        let obj = val.as_object().ok_or_else(|| {
            TokenizerError::InvalidConfig("post_processor must be null or an object".to_string())
        })?;
        let ty = obj.get("type").and_then(Value::as_str).ok_or_else(|| {
            TokenizerError::InvalidConfig("post_processor missing string `type`".to_string())
        })?;

        match ty {
            "ByteLevel" => Ok(Self::default()),
            "Sequence" => {
                let procs = obj
                    .get("processors")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        TokenizerError::InvalidConfig(
                            "Sequence post_processor missing `processors` array".to_string(),
                        )
                    })?;
                let mut combined = Self::default();
                for child_val in procs {
                    let child = Self::from_value(Some(child_val))?;
                    if !child.prefix_ids.is_empty() {
                        let mut new_prefix = child.prefix_ids;
                        new_prefix.extend_from_slice(&combined.prefix_ids);
                        combined.prefix_ids = new_prefix;
                    }
                    combined.suffix_ids.extend_from_slice(&child.suffix_ids);
                }
                Ok(combined)
            }
            "TemplateProcessing" => parse_template_processing(obj),
            other => Err(TokenizerError::Unsupported(format!(
                "unsupported post_processor type: {other}"
            ))),
        }
    }
}

fn parse_template_processing(
    obj: &serde_json::Map<String, Value>,
) -> Result<PostProcessor, TokenizerError> {
    let single = obj.get("single").and_then(Value::as_array).ok_or_else(|| {
        TokenizerError::InvalidConfig(
            "TemplateProcessing missing `single` template array".to_string(),
        )
    })?;

    let special_map: HashMap<String, TemplateSpecialTokenJson> = match obj.get("special_tokens") {
        Some(v) if !v.is_null() => serde_json::from_value(v.clone())?,
        _ => HashMap::new(),
    };

    let mut prefix_ids = Vec::new();
    let mut suffix_ids = Vec::new();
    let mut seen_sequence_a = false;

    for piece_val in single {
        let piece_obj = piece_val.as_object().ok_or_else(|| {
            TokenizerError::InvalidConfig(
                "TemplateProcessing `single` entry must be an object".to_string(),
            )
        })?;
        if let Some(seq) = piece_obj.get("Sequence") {
            let id = seq.get("id").and_then(Value::as_str).ok_or_else(|| {
                TokenizerError::InvalidConfig(
                    "TemplateProcessing Sequence missing `id`".to_string(),
                )
            })?;
            if id != "A" || seen_sequence_a {
                return Err(TokenizerError::Unsupported(format!(
                    "unsupported TemplateProcessing Sequence `{id}` in `single`"
                )));
            }
            seen_sequence_a = true;
        } else if let Some(spec) = piece_obj.get("SpecialToken") {
            let id = spec.get("id").and_then(Value::as_str).ok_or_else(|| {
                TokenizerError::InvalidConfig(
                    "TemplateProcessing SpecialToken missing `id`".to_string(),
                )
            })?;
            let entry = special_map.get(id).ok_or_else(|| {
                TokenizerError::MissingVocabToken(format!(
                    "TemplateProcessing special token `{id}` missing in `special_tokens`"
                ))
            })?;
            if entry.ids.is_empty() {
                return Err(TokenizerError::InvalidConfig(format!(
                    "TemplateProcessing special token `{id}` has empty `ids`"
                )));
            }
            if seen_sequence_a {
                suffix_ids.extend_from_slice(&entry.ids);
            } else {
                prefix_ids.extend_from_slice(&entry.ids);
            }
        } else {
            return Err(TokenizerError::Unsupported(
                "unsupported TemplateProcessing piece in `single`".to_string(),
            ));
        }
    }

    if !seen_sequence_a {
        return Err(TokenizerError::Unsupported(
            "TemplateProcessing `single` template without Sequence `A` is unsupported".to_string(),
        ));
    }

    Ok(PostProcessor {
        prefix_ids,
        suffix_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::unicode_lut::{unicode_props, CLS_MASK, CLS_N, FLAG_HAN};

    #[test]
    fn unicode_lut_spot_checks() {
        assert_eq!(unicode_props('\u{11C67}') & CLS_MASK, CLS_N);
        assert_ne!(unicode_props('漢') & FLAG_HAN, 0);
    }
}
