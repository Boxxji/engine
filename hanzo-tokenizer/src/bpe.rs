// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: MIT

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use rustc_hash::{FxBuildHasher, FxHashMap};
use serde::Deserialize;
use serde_json::Value;

use crate::TokenizerError;

const EMPTY_RANK: u32 = u32::MAX;
const NO_LINK: usize = usize::MAX;
const CACHE_SLOTS: usize = 2048;
const CACHE_MASK: usize = CACHE_SLOTS - 1;
const CACHE_MAX_PIECE_BYTES: usize = 64;
const CACHE_MAX_OUT_IDS: usize = 16;

static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) const BYTE_TO_CHAR: [char; 256] = build_byte_to_char();
pub(crate) const BYTE_TO_UTF8: [([u8; 2], u8); 256] = build_byte_to_utf8();

const fn build_byte_to_char() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut is_direct = [false; 256];

    let mut b = b'!' as usize;
    while b <= b'~' as usize {
        is_direct[b] = true;
        b += 1;
    }
    b = 0xA1;
    while b <= 0xAC {
        is_direct[b] = true;
        b += 1;
    }
    b = 0xAE;
    while b <= 0xFF {
        is_direct[b] = true;
        b += 1;
    }

    let mut extra = 0u32;
    let mut i = 0usize;
    while i < 256 {
        let cp = if is_direct[i] {
            i as u32
        } else {
            let v = 256 + extra;
            extra += 1;
            v
        };
        table[i] = match char::from_u32(cp) {
            Some(c) => c,
            None => '\0',
        };
        i += 1;
    }
    table
}

const fn build_byte_to_utf8() -> [([u8; 2], u8); 256] {
    let mut out = [([0u8; 2], 0u8); 256];
    let mut i = 0usize;
    while i < 256 {
        let cp = BYTE_TO_CHAR[i] as u32;
        if cp < 0x80 {
            out[i] = ([cp as u8, 0], 1);
        } else {
            let b0 = 0xC0 | ((cp >> 6) as u8);
            let b1 = 0x80 | ((cp & 0x3F) as u8);
            out[i] = ([b0, b1], 2);
        }
        i += 1;
    }
    out
}

#[derive(Clone, Copy)]
struct MergeSlot {
    key: u64,
    rank: u32,
    merged_id: u32,
}

#[derive(Clone)]
struct MergeTable {
    slots: Vec<MergeSlot>,
    mask: usize,
}

impl MergeTable {
    fn from_pairs(pairs: Vec<(u32, u32, u32, u32)>) -> Self {
        if pairs.is_empty() {
            return Self {
                slots: Vec::new(),
                mask: 0,
            };
        }
        let min_cap = (pairs.len().saturating_mul(10) / 7).max(8);
        let cap = min_cap.next_power_of_two();
        let mask = cap - 1;
        let mut slots = vec![
            MergeSlot {
                key: 0,
                rank: EMPTY_RANK,
                merged_id: 0,
            };
            cap
        ];

        for (a, b, rank, merged_id) in pairs {
            let key = pack_pair(a, b);
            let mut idx = hash_pair(key) & mask;
            loop {
                let slot = &mut slots[idx];
                if slot.rank == EMPTY_RANK || slot.key == key {
                    slot.key = key;
                    slot.rank = rank;
                    slot.merged_id = merged_id;
                    break;
                }
                idx = (idx + 1) & mask;
            }
        }

        Self { slots, mask }
    }

    #[inline(always)]
    fn get(&self, a: u32, b: u32) -> Option<(u32, u32)> {
        if self.slots.is_empty() {
            return None;
        }
        let key = pack_pair(a, b);
        let mut idx = hash_pair(key) & self.mask;
        loop {
            // Safety: idx is masked to `self.slots.len() - 1` where len is a non-zero power of 2.
            let slot = self.slots[idx];
            if slot.rank == EMPTY_RANK {
                return None;
            }
            if slot.key == key {
                return Some((slot.rank, slot.merged_id));
            }
            idx = (idx + 1) & self.mask;
        }
    }
}

#[inline(always)]
const fn pack_pair(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | (b as u64)
}

#[inline(always)]
const fn hash_pair(key: u64) -> usize {
    let x = key.wrapping_mul(0x517c_c1b7_2722_0a95);
    (x ^ (x >> 32)) as usize
}

#[derive(Clone, Copy)]
struct ThreadCacheSlot {
    model_id: u64,
    hash: u64,
    /// Low 15 bits = byte length; bit 15 = is_byte_level.
    meta: u16,
    out_len: u8,
    piece: [u8; CACHE_MAX_PIECE_BYTES],
    out_ids: [u32; CACHE_MAX_OUT_IDS],
}

impl Default for ThreadCacheSlot {
    fn default() -> Self {
        Self {
            model_id: 0,
            hash: 0,
            meta: 0,
            out_len: 0,
            piece: [0; CACHE_MAX_PIECE_BYTES],
            out_ids: [0; CACHE_MAX_OUT_IDS],
        }
    }
}

thread_local! {
    static PRETOKEN_CACHE: RefCell<Box<[ThreadCacheSlot; CACHE_SLOTS]>> =
        RefCell::new(vec![ThreadCacheSlot::default(); CACHE_SLOTS].into_boxed_slice().try_into().unwrap_or_else(|_| unreachable!()));
}

#[inline(always)]
fn hash_piece(model_id: u64, is_byte_level: bool, bytes: &[u8]) -> u64 {
    let mut h = model_id ^ ((is_byte_level as u64) << 63);
    for &b in bytes {
        h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(b as u64);
    }
    h ^ (h >> 33)
}

#[derive(Deserialize)]
struct BpeModelConfig {
    #[serde(rename = "type")]
    _model_type: String,
    #[serde(default)]
    dropout: Option<f32>,
    #[serde(default)]
    unk_token: Option<String>,
    #[serde(default)]
    continuing_subword_prefix: Option<String>,
    #[serde(default)]
    end_of_word_suffix: Option<String>,
    #[serde(default)]
    fuse_unk: bool,
    #[serde(default)]
    byte_fallback: bool,
    #[serde(default)]
    ignore_merges: bool,
    vocab: FxHashMap<String, Value>,
    #[serde(default)]
    merges: Vec<Value>,
}

#[derive(Clone)]
pub(crate) struct BpeModel {
    model_id: u64,
    vocab: FxHashMap<Box<str>, u32>,
    merges: MergeTable,
    byte_to_id: [Option<u32>; 256],
    byte_fallback_ids: [Option<u32>; 256],
    unk_token: Option<String>,
    unk_id: Option<u32>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    fuse_unk: bool,
    byte_fallback: bool,
    ignore_merges: bool,
}

impl std::fmt::Debug for BpeModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BpeModel")
            .field("vocab_size", &self.vocab.len())
            .field("ignore_merges", &self.ignore_merges)
            .field("byte_fallback", &self.byte_fallback)
            .finish()
    }
}

impl BpeModel {
    pub(crate) fn from_value(value: &Value) -> Result<Self, TokenizerError> {
        let obj = value.as_object().ok_or_else(|| {
            TokenizerError::InvalidConfig("model must be a JSON object".to_string())
        })?;
        let ty = obj.get("type").and_then(Value::as_str).ok_or_else(|| {
            TokenizerError::InvalidConfig("model missing string `type`".to_string())
        })?;
        if ty != "BPE" {
            return Err(TokenizerError::Unsupported(format!(
                "unsupported model type: {ty}"
            )));
        }

        let cfg: BpeModelConfig = serde_json::from_value(value.clone())?;
        if let Some(d) = cfg.dropout {
            if !d.is_finite() || d.abs() > 0.0 {
                return Err(TokenizerError::Unsupported(
                    "BPE dropout is not supported".to_string(),
                ));
            }
        }

        let continuing_subword_prefix = cfg.continuing_subword_prefix.filter(|s| !s.is_empty());
        let end_of_word_suffix = cfg.end_of_word_suffix.filter(|s| !s.is_empty());
        let prefix_len = continuing_subword_prefix.as_ref().map_or(0, String::len);

        let mut vocab = FxHashMap::with_capacity_and_hasher(cfg.vocab.len(), FxBuildHasher);
        for (tok, id_val) in cfg.vocab {
            let id_u64 = id_val.as_u64().ok_or_else(|| {
                TokenizerError::InvalidConfig(format!("vocab id for `{tok}` must be an integer"))
            })?;
            let id = u32::try_from(id_u64).map_err(|_| {
                TokenizerError::InvalidConfig(format!("vocab id for `{tok}` exceeds u32"))
            })?;
            vocab.insert(tok.into_boxed_str(), id);
        }

        if cfg.merges.len() >= u32::MAX as usize {
            return Err(TokenizerError::InvalidConfig(
                "merges length exceeds u32 capacity".to_string(),
            ));
        }

        let mut pairs = Vec::with_capacity(cfg.merges.len());
        for (rank_idx, entry) in cfg.merges.iter().enumerate() {
            let (a, b): (&str, &str) = match entry {
                Value::String(s) => {
                    let mut parts = s.split(' ');
                    match (parts.next(), parts.next(), parts.next()) {
                        (Some(left), Some(right), None) => (left, right),
                        _ => {
                            return Err(TokenizerError::InvalidConfig(format!(
                                "invalid string merge entry at rank {rank_idx}: `{s}`"
                            )));
                        }
                    }
                }
                Value::Array(arr) => {
                    if arr.len() != 2 {
                        return Err(TokenizerError::InvalidConfig(format!(
                            "invalid array merge entry length at rank {rank_idx}"
                        )));
                    }
                    let left = arr[0].as_str().ok_or_else(|| {
                        TokenizerError::InvalidConfig(format!(
                            "non-string merge token at rank {rank_idx}"
                        ))
                    })?;
                    let right = arr[1].as_str().ok_or_else(|| {
                        TokenizerError::InvalidConfig(format!(
                            "non-string merge token at rank {rank_idx}"
                        ))
                    })?;
                    (left, right)
                }
                _ => {
                    return Err(TokenizerError::InvalidConfig(format!(
                        "invalid merge entry type at rank {rank_idx}"
                    )));
                }
            };

            let a_id = vocab
                .get(a)
                .copied()
                .ok_or_else(|| TokenizerError::MissingVocabToken(a.to_string()))?;
            let b_id = vocab
                .get(b)
                .copied()
                .ok_or_else(|| TokenizerError::MissingVocabToken(b.to_string()))?;

            let b_tail = b.get(prefix_len..).ok_or_else(|| {
                TokenizerError::InvalidConfig(format!(
                    "merge token `{b}` incompatible with continuing_subword_prefix"
                ))
            })?;
            let mut merged = String::with_capacity(a.len() + b_tail.len());
            merged.push_str(a);
            merged.push_str(b_tail);
            let merged_id = vocab
                .get(merged.as_str())
                .copied()
                .ok_or(TokenizerError::MissingVocabToken(merged))?;

            pairs.push((a_id, b_id, rank_idx as u32, merged_id));
        }

        let merges = MergeTable::from_pairs(pairs);

        let mut byte_to_id = [None; 256];
        for b in 0..=255u8 {
            let (utf8, len) = BYTE_TO_UTF8[b as usize];
            let s = std::str::from_utf8(&utf8[..len as usize])
                .map_err(|_| TokenizerError::InvalidConfig("invalid UTF-8 LUT".to_string()))?;
            byte_to_id[b as usize] = vocab.get(s).copied();
        }

        let mut byte_fallback_ids = [None; 256];
        if cfg.byte_fallback {
            for b in 0..=255u8 {
                let code = format!("<{b:#04X}>");
                byte_fallback_ids[b as usize] = vocab.get(code.as_str()).copied();
            }
        }

        let unk_id = cfg
            .unk_token
            .as_deref()
            .and_then(|tok| vocab.get(tok).copied());

        Ok(Self {
            model_id: NEXT_MODEL_ID.fetch_add(1, AtomicOrdering::Relaxed),
            vocab,
            merges,
            byte_to_id,
            byte_fallback_ids,
            unk_token: cfg.unk_token,
            unk_id,
            continuing_subword_prefix,
            end_of_word_suffix,
            fuse_unk: cfg.fuse_unk,
            byte_fallback: cfg.byte_fallback,
            ignore_merges: cfg.ignore_merges,
        })
    }

    #[inline(always)]
    pub(crate) fn has_affixes(&self) -> bool {
        self.continuing_subword_prefix.is_some() || self.end_of_word_suffix.is_some()
    }

    /// Encode a raw pre-tokenized UTF-8 slice as a ByteLevel piece (each raw byte maps directly
    /// through the GPT-2 byte alphabet without allocating an intermediate ByteLevel string).
    pub(crate) fn encode_byte_level_piece(
        &self,
        raw: &str,
        out: &mut Vec<u32>,
    ) -> Result<(), TokenizerError> {
        if raw.is_empty() {
            return Ok(());
        }
        if self.has_affixes() {
            let mapped = encode_byte_level_string(raw.as_bytes());
            return self.encode_piece(&mapped, out);
        }

        let bytes = raw.as_bytes();
        if self.ignore_merges {
            if let Some(whole_id) = self.lookup_byte_level_whole_word(bytes) {
                out.push(whole_id);
                return Ok(());
            }
        }

        if bytes.len() <= CACHE_MAX_PIECE_BYTES {
            let h = hash_piece(self.model_id, true, bytes);
            let slot_idx = (h as usize) & CACHE_MASK;
            let meta = (bytes.len() as u16) | 0x8000;
            let hit = PRETOKEN_CACHE.with(|cell| {
                let slots = cell.borrow();
                let slot = &slots[slot_idx];
                if slot.model_id == self.model_id
                    && slot.hash == h
                    && slot.meta == meta
                    && &slot.piece[..bytes.len()] == bytes
                {
                    out.extend_from_slice(&slot.out_ids[..slot.out_len as usize]);
                    true
                } else {
                    false
                }
            });
            if hit {
                return Ok(());
            }

            let start_len = out.len();
            self.encode_byte_level_uncached(bytes, out)?;
            let produced = &out[start_len..];
            if produced.len() <= CACHE_MAX_OUT_IDS {
                PRETOKEN_CACHE.with(|cell| {
                    let mut slots = cell.borrow_mut();
                    let slot = &mut slots[slot_idx];
                    slot.model_id = self.model_id;
                    slot.hash = h;
                    slot.meta = meta;
                    slot.out_len = produced.len() as u8;
                    slot.piece[..bytes.len()].copy_from_slice(bytes);
                    slot.out_ids[..produced.len()].copy_from_slice(produced);
                });
            }
            return Ok(());
        }

        self.encode_byte_level_uncached(bytes, out)
    }

    /// Encode an already pre-tokenized piece directly against `vocab` and `merges`.
    pub(crate) fn encode_piece(
        &self,
        piece: &str,
        out: &mut Vec<u32>,
    ) -> Result<(), TokenizerError> {
        if piece.is_empty() {
            return Ok(());
        }
        if self.ignore_merges {
            if let Some(&id) = self.vocab.get(piece) {
                out.push(id);
                return Ok(());
            }
        }

        let bytes = piece.as_bytes();
        if bytes.len() <= CACHE_MAX_PIECE_BYTES {
            let h = hash_piece(self.model_id, false, bytes);
            let slot_idx = (h as usize) & CACHE_MASK;
            let meta = bytes.len() as u16;
            let hit = PRETOKEN_CACHE.with(|cell| {
                let slots = cell.borrow();
                let slot = &slots[slot_idx];
                if slot.model_id == self.model_id
                    && slot.hash == h
                    && slot.meta == meta
                    && &slot.piece[..bytes.len()] == bytes
                {
                    out.extend_from_slice(&slot.out_ids[..slot.out_len as usize]);
                    true
                } else {
                    false
                }
            });
            if hit {
                return Ok(());
            }

            let start_len = out.len();
            self.encode_piece_uncached(piece, out)?;
            let produced = &out[start_len..];
            if produced.len() <= CACHE_MAX_OUT_IDS {
                PRETOKEN_CACHE.with(|cell| {
                    let mut slots = cell.borrow_mut();
                    let slot = &mut slots[slot_idx];
                    slot.model_id = self.model_id;
                    slot.hash = h;
                    slot.meta = meta;
                    slot.out_len = produced.len() as u8;
                    slot.piece[..bytes.len()].copy_from_slice(bytes);
                    slot.out_ids[..produced.len()].copy_from_slice(produced);
                });
            }
            return Ok(());
        }

        self.encode_piece_uncached(piece, out)
    }

    fn lookup_byte_level_whole_word(&self, raw: &[u8]) -> Option<u32> {
        if raw.len() <= 256 {
            let mut buf = [0u8; 512];
            let mut n = 0usize;
            for &b in raw {
                let (utf8, len) = BYTE_TO_UTF8[b as usize];
                buf[n] = utf8[0];
                if len == 2 {
                    buf[n + 1] = utf8[1];
                }
                n += len as usize;
            }
            let s = std::str::from_utf8(&buf[..n]).ok()?;
            self.vocab.get(s).copied()
        } else {
            let s = encode_byte_level_string(raw);
            self.vocab.get(s.as_str()).copied()
        }
    }

    fn encode_byte_level_uncached(
        &self,
        bytes: &[u8],
        out: &mut Vec<u32>,
    ) -> Result<(), TokenizerError> {
        if bytes.len() <= 32 {
            let mut ids = [0u32; 32];
            let mut count = 0usize;
            for &b in bytes {
                if let Some(id) = self.byte_to_id[b as usize] {
                    ids[count] = id;
                    count += 1;
                } else {
                    let s = encode_byte_level_string(bytes);
                    return self.encode_piece_uncached(&s, out);
                }
            }
            self.merge_small_stack(&mut ids, count, out);
            return Ok(());
        }

        let mut init = Vec::with_capacity(bytes.len());
        for &b in bytes {
            if let Some(id) = self.byte_to_id[b as usize] {
                init.push(id);
            } else {
                let s = encode_byte_level_string(bytes);
                return self.encode_piece_uncached(&s, out);
            }
        }
        self.merge_large_heap(&init, out);
        Ok(())
    }

    fn encode_piece_uncached(&self, piece: &str, out: &mut Vec<u32>) -> Result<(), TokenizerError> {
        let mut init = Vec::with_capacity(piece.chars().count());
        let mut pending_unk = false;
        let mut scratch = String::new();

        let mut iter = piece.char_indices().peekable();
        while let Some((start, _ch)) = iter.next() {
            let end = iter.peek().map_or(piece.len(), |&(idx, _)| idx);
            let is_first = start == 0;
            let is_last = iter.peek().is_none();
            let base = &piece[start..end];

            let lookup: &str = if (!is_first && self.continuing_subword_prefix.is_some())
                || (is_last && self.end_of_word_suffix.is_some())
            {
                scratch.clear();
                if !is_first {
                    if let Some(ref prefix) = self.continuing_subword_prefix {
                        scratch.push_str(prefix);
                    }
                }
                scratch.push_str(base);
                if is_last {
                    if let Some(ref suffix) = self.end_of_word_suffix {
                        scratch.push_str(suffix);
                    }
                }
                scratch.as_str()
            } else {
                base
            };

            if let Some(&id) = self.vocab.get(lookup) {
                if pending_unk {
                    init.push(self.resolve_unk_id()?);
                    pending_unk = false;
                }
                init.push(id);
            } else if self.byte_fallback
                && lookup
                    .as_bytes()
                    .iter()
                    .all(|&b| self.byte_fallback_ids[b as usize].is_some())
            {
                if pending_unk {
                    init.push(self.resolve_unk_id()?);
                    pending_unk = false;
                }
                for &b in lookup.as_bytes() {
                    if let Some(bid) = self.byte_fallback_ids[b as usize] {
                        init.push(bid);
                    }
                }
            } else if self.unk_token.is_some() {
                if self.fuse_unk {
                    pending_unk = true;
                } else {
                    init.push(self.resolve_unk_id()?);
                }
            }
        }
        if pending_unk {
            init.push(self.resolve_unk_id()?);
        }

        if init.len() <= 32 {
            let mut ids = [0u32; 32];
            ids[..init.len()].copy_from_slice(&init);
            self.merge_small_stack(&mut ids, init.len(), out);
        } else {
            self.merge_large_heap(&init, out);
        }
        Ok(())
    }

    fn resolve_unk_id(&self) -> Result<u32, TokenizerError> {
        if let Some(id) = self.unk_id {
            return Ok(id);
        }
        let tok = self.unk_token.clone().unwrap_or_default();
        Err(TokenizerError::MissingVocabToken(tok))
    }

    #[inline]
    fn merge_small_stack(&self, ids: &mut [u32; 32], mut len: usize, out: &mut Vec<u32>) {
        if len == 0 {
            return;
        }
        if len == 1 {
            out.push(ids[0]);
            return;
        }
        if len == 2 {
            if let Some((_, merged)) = self.merges.get(ids[0], ids[1]) {
                out.push(merged);
            } else {
                out.push(ids[0]);
                out.push(ids[1]);
            }
            return;
        }

        let mut pair_rank = [EMPTY_RANK; 31];
        let mut pair_new = [0u32; 31];
        for i in 0..(len - 1) {
            if let Some((r, m)) = self.merges.get(ids[i], ids[i + 1]) {
                pair_rank[i] = r;
                pair_new[i] = m;
            }
        }

        while len > 1 {
            let mut best_rank = EMPTY_RANK;
            let mut best_idx = 0usize;
            for (i, &r) in pair_rank[..(len - 1)].iter().enumerate() {
                if r < best_rank {
                    best_rank = r;
                    best_idx = i;
                }
            }
            if best_rank == EMPTY_RANK {
                break;
            }

            ids[best_idx] = pair_new[best_idx];
            let tail_start = best_idx + 1;
            if tail_start + 1 < len {
                ids.copy_within((tail_start + 1)..len, tail_start);
            }
            if best_idx + 1 < len - 1 {
                pair_rank.copy_within((best_idx + 1)..(len - 1), best_idx);
                pair_new.copy_within((best_idx + 1)..(len - 1), best_idx);
            }
            len -= 1;

            if best_idx > 0 {
                let (r, m) = self
                    .merges
                    .get(ids[best_idx - 1], ids[best_idx])
                    .unwrap_or((EMPTY_RANK, 0));
                pair_rank[best_idx - 1] = r;
                pair_new[best_idx - 1] = m;
            }
            if best_idx + 1 < len {
                let (r, m) = self
                    .merges
                    .get(ids[best_idx], ids[best_idx + 1])
                    .unwrap_or((EMPTY_RANK, 0));
                pair_rank[best_idx] = r;
                pair_new[best_idx] = m;
            }
        }

        out.extend_from_slice(&ids[..len]);
    }

    fn merge_large_heap(&self, init: &[u32], out: &mut Vec<u32>) {
        if init.is_empty() {
            return;
        }
        #[derive(Clone, Copy)]
        struct Symbol {
            id: u32,
            prev: usize,
            next: usize,
            alive: bool,
        }
        #[derive(Clone, Copy, Eq, PartialEq)]
        struct MergeCand {
            rank: u32,
            pos: usize,
            new_id: u32,
        }
        impl Ord for MergeCand {
            fn cmp(&self, other: &Self) -> Ordering {
                other
                    .rank
                    .cmp(&self.rank)
                    .then_with(|| other.pos.cmp(&self.pos))
            }
        }
        impl PartialOrd for MergeCand {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let n = init.len();
        let mut syms: Vec<Symbol> = Vec::with_capacity(n);
        for (i, &id) in init.iter().enumerate() {
            syms.push(Symbol {
                id,
                prev: if i == 0 { NO_LINK } else { i - 1 },
                next: if i + 1 == n { NO_LINK } else { i + 1 },
                alive: true,
            });
        }

        let mut heap = BinaryHeap::with_capacity(n);
        for i in 0..(n - 1) {
            if let Some((rank, new_id)) = self.merges.get(syms[i].id, syms[i + 1].id) {
                heap.push(MergeCand {
                    rank,
                    pos: i,
                    new_id,
                });
            }
        }

        while let Some(top) = heap.pop() {
            if !syms[top.pos].alive {
                continue;
            }
            let right_pos = syms[top.pos].next;
            if right_pos == NO_LINK || !syms[right_pos].alive {
                continue;
            }
            let Some((cur_rank, cur_new)) = self.merges.get(syms[top.pos].id, syms[right_pos].id)
            else {
                continue;
            };
            if cur_rank != top.rank || cur_new != top.new_id {
                continue;
            }

            syms[top.pos].id = top.new_id;
            let next_next = syms[right_pos].next;
            syms[top.pos].next = next_next;
            syms[right_pos].alive = false;
            if next_next != NO_LINK {
                syms[next_next].prev = top.pos;
            }

            let prev_pos = syms[top.pos].prev;
            if prev_pos != NO_LINK {
                if let Some((rank, new_id)) = self.merges.get(syms[prev_pos].id, syms[top.pos].id) {
                    heap.push(MergeCand {
                        rank,
                        pos: prev_pos,
                        new_id,
                    });
                }
            }
            if next_next != NO_LINK {
                if let Some((rank, new_id)) = self.merges.get(syms[top.pos].id, syms[next_next].id)
                {
                    heap.push(MergeCand {
                        rank,
                        pos: top.pos,
                        new_id,
                    });
                }
            }
        }

        let mut cur = 0usize;
        while cur != NO_LINK {
            out.push(syms[cur].id);
            cur = syms[cur].next;
        }
    }
}

pub(crate) fn encode_byte_level_string(raw: &[u8]) -> String {
    let mut bytes = Vec::with_capacity(raw.len() * 2);
    for &b in raw {
        let (utf8, len) = BYTE_TO_UTF8[b as usize];
        bytes.push(utf8[0]);
        if len == 2 {
            bytes.push(utf8[1]);
        }
    }
    // Safety: BYTE_TO_UTF8 entries are valid 1- or 2-byte UTF-8 sequences.
    String::from_utf8(bytes).unwrap_or_default()
}
