// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: MIT

use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::{Tokenizer, TokenizerError};

const CONSECUTIVE_MISS_THRESHOLD: usize = 2;
const COOLDOWN_CALLS: usize = 8;
const MAX_PREFIX_SCAN_ENTRIES: usize = 32;

/// Runtime statistics for [`PromptTokenCache`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PromptCacheStats {
    pub exact_hits: u64,
    pub prefix_hits: u64,
    pub misses: u64,
    pub bypassed: u64,
}

#[derive(Clone)]
struct CacheEntry {
    tokenizer_id: u64,
    add_special_tokens: bool,
    text_hash: u64,
    text: String,
    tokens: Vec<u32>,
    /// Safe `\n` split checkpoints `(byte_offset, token_end_index_before_post_eos)`.
    checkpoints: Vec<(usize, usize)>,
}

struct CacheSlot {
    entry: CacheEntry,
    prev: Option<usize>,
    next: Option<usize>,
    hash_next: Option<usize>,
}

/// Bounded LRU prompt token cache with boundary-safe incremental `\n` prefix reuse and
/// online self-bypass protection against non-incremental workloads.
pub struct PromptTokenCache {
    max_entries: usize,
    stats: PromptCacheStats,
    consecutive_misses: usize,
    cooldown_remaining: usize,
    slots: Vec<CacheSlot>,
    head: Option<usize>,
    tail: Option<usize>,
    exact_index: FxHashMap<u64, usize>,
}

impl std::fmt::Debug for PromptTokenCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PromptTokenCache")
            .field("max_entries", &self.max_entries)
            .field("len", &self.slots.len())
            .field("stats", &self.stats)
            .finish()
    }
}

impl PromptTokenCache {
    /// Creates a new prompt token cache retaining up to `max_entries` recent prompts.
    pub fn new(max_entries: usize) -> Self {
        Self {
            max_entries,
            stats: PromptCacheStats::default(),
            consecutive_misses: 0,
            cooldown_remaining: 0,
            slots: Vec::with_capacity(max_entries.min(64)),
            head: None,
            tail: None,
            exact_index: FxHashMap::with_capacity_and_hasher(max_entries.min(64), FxBuildHasher),
        }
    }

    /// Returns cumulative hit/miss/bypass statistics for this cache.
    pub fn stats(&self) -> PromptCacheStats {
        self.stats
    }

    pub(crate) fn encode_with(
        &mut self,
        tokenizer: &Tokenizer,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, TokenizerError> {
        if self.max_entries == 0 {
            self.stats.bypassed += 1;
            return tokenizer.encode_with_special_tokens(text, add_special_tokens);
        }

        if self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
            self.stats.bypassed += 1;
            return tokenizer.encode_with_special_tokens(text, add_special_tokens);
        }

        let tokenizer_id = tokenizer.id();
        let text_hash = hash_prompt_key(tokenizer_id, add_special_tokens, text.as_bytes());

        if let Some(slot_idx) = self.find_exact(tokenizer_id, add_special_tokens, text_hash, text) {
            self.move_to_front(slot_idx);
            self.consecutive_misses = 0;
            self.stats.exact_hits += 1;
            return Ok(self.slots[slot_idx].entry.tokens.clone());
        }

        if tokenizer.supports_newline_checkpoints() {
            if let Some((slot_idx, split_byte, tok_end)) =
                self.find_longest_prefix_checkpoint(tokenizer_id, add_special_tokens, text)
            {
                self.move_to_front(slot_idx);
                let mut full_tokens = Vec::new();
                let mut full_checkpoints = Vec::new();
                {
                    let cached = &self.slots[slot_idx].entry;
                    full_tokens.extend_from_slice(&cached.tokens[..tok_end]);
                    for &(cp_byte, cp_tok) in &cached.checkpoints {
                        if cp_byte <= split_byte {
                            full_checkpoints.push((cp_byte, cp_tok));
                        } else {
                            break;
                        }
                    }
                }

                let tail = &text[split_byte..];
                let (tail_tokens, tail_checkpoints) = tokenizer.encode_incremental_tail(
                    tail,
                    add_special_tokens,
                    split_byte,
                    tok_end,
                )?;
                full_tokens.extend_from_slice(&tail_tokens);
                full_checkpoints.extend_from_slice(&tail_checkpoints);

                self.consecutive_misses = 0;
                self.stats.prefix_hits += 1;
                self.insert(CacheEntry {
                    tokenizer_id,
                    add_special_tokens,
                    text_hash,
                    text: text.to_owned(),
                    tokens: full_tokens.clone(),
                    checkpoints: full_checkpoints,
                });
                return Ok(full_tokens);
            }
        }

        self.stats.misses += 1;
        self.consecutive_misses += 1;
        if self.consecutive_misses >= CONSECUTIVE_MISS_THRESHOLD {
            self.cooldown_remaining = COOLDOWN_CALLS;
        }

        let (tokens, checkpoints) = tokenizer.encode_with_checkpoints(text, add_special_tokens)?;
        self.insert(CacheEntry {
            tokenizer_id,
            add_special_tokens,
            text_hash,
            text: text.to_owned(),
            tokens: tokens.clone(),
            checkpoints,
        });
        Ok(tokens)
    }

    fn find_exact(
        &self,
        tokenizer_id: u64,
        add_special_tokens: bool,
        text_hash: u64,
        text: &str,
    ) -> Option<usize> {
        let mut cur = self.exact_index.get(&text_hash).copied();
        while let Some(idx) = cur {
            let slot = &self.slots[idx];
            if slot.entry.tokenizer_id == tokenizer_id
                && slot.entry.add_special_tokens == add_special_tokens
                && slot.entry.text == text
            {
                return Some(idx);
            }
            cur = slot.hash_next;
        }
        None
    }

    fn find_longest_prefix_checkpoint(
        &self,
        tokenizer_id: u64,
        add_special_tokens: bool,
        text: &str,
    ) -> Option<(usize, usize, usize)> {
        let tb = text.as_bytes();
        if tb.len() < 2 {
            return None;
        }
        let mut best: Option<(usize, usize, usize)> = None;
        let mut cur = self.head;
        let mut scanned = 0usize;

        while let Some(idx) = cur {
            if scanned >= MAX_PREFIX_SCAN_ENTRIES {
                break;
            }
            scanned += 1;
            let slot = &self.slots[idx];
            let entry = &slot.entry;
            if entry.tokenizer_id == tokenizer_id
                && entry.add_special_tokens == add_special_tokens
                && !entry.checkpoints.is_empty()
            {
                let eb = entry.text.as_bytes();
                let best_so_far = best.map_or(0, |(_, b, _)| b);

                for &(cp_byte, tok_end) in entry.checkpoints.iter().rev() {
                    if cp_byte <= best_so_far {
                        break;
                    }
                    if cp_byte >= tb.len() {
                        continue;
                    }
                    if !is_safe_checkpoint_follower(tb[cp_byte]) {
                        continue;
                    }
                    if cp_byte >= 8 && tb[cp_byte - 8..cp_byte] != eb[cp_byte - 8..cp_byte] {
                        continue;
                    }
                    let head_chk = cp_byte.min(8);
                    if tb[..head_chk] != eb[..head_chk] {
                        continue;
                    }
                    if tb[..cp_byte] == eb[..cp_byte] {
                        best = Some((idx, cp_byte, tok_end));
                        break;
                    }
                }
            }
            cur = slot.next;
        }

        best
    }

    fn move_to_front(&mut self, idx: usize) {
        if self.head == Some(idx) {
            return;
        }
        self.unlink_lru(idx);
        self.push_front_lru(idx);
    }

    fn unlink_lru(&mut self, idx: usize) {
        let prev = self.slots[idx].prev;
        let next = self.slots[idx].next;
        match prev {
            Some(p) => self.slots[p].next = next,
            None => self.head = next,
        }
        match next {
            Some(n) => self.slots[n].prev = prev,
            None => self.tail = prev,
        }
        self.slots[idx].prev = None;
        self.slots[idx].next = None;
    }

    fn push_front_lru(&mut self, idx: usize) {
        let old_head = self.head;
        self.slots[idx].prev = None;
        self.slots[idx].next = old_head;
        if let Some(h) = old_head {
            self.slots[h].prev = Some(idx);
        } else {
            self.tail = Some(idx);
        }
        self.head = Some(idx);
    }

    fn insert(&mut self, entry: CacheEntry) {
        if self.max_entries == 0 {
            return;
        }
        let new_hash = entry.text_hash;
        if self.slots.len() < self.max_entries {
            let idx = self.slots.len();
            let old_bucket_head = self.exact_index.insert(new_hash, idx);
            self.slots.push(CacheSlot {
                entry,
                prev: None,
                next: None,
                hash_next: old_bucket_head,
            });
            self.push_front_lru(idx);
        } else if let Some(evict_idx) = self.tail {
            let old_hash = self.slots[evict_idx].entry.text_hash;
            self.remove_from_hash_bucket(old_hash, evict_idx);
            self.unlink_lru(evict_idx);
            let old_bucket_head = self.exact_index.insert(new_hash, evict_idx);
            self.slots[evict_idx].entry = entry;
            self.slots[evict_idx].hash_next = old_bucket_head;
            self.push_front_lru(evict_idx);
        }
    }

    fn remove_from_hash_bucket(&mut self, hash: u64, target_idx: usize) {
        let Some(head_idx) = self.exact_index.get(&hash).copied() else {
            return;
        };
        if head_idx == target_idx {
            if let Some(next_idx) = self.slots[target_idx].hash_next {
                self.exact_index.insert(hash, next_idx);
            } else {
                self.exact_index.remove(&hash);
            }
            self.slots[target_idx].hash_next = None;
            return;
        }
        let mut prev_idx = head_idx;
        while let Some(cur_idx) = self.slots[prev_idx].hash_next {
            if cur_idx == target_idx {
                self.slots[prev_idx].hash_next = self.slots[target_idx].hash_next;
                self.slots[target_idx].hash_next = None;
                return;
            }
            prev_idx = cur_idx;
        }
    }
}

#[inline(always)]
pub(crate) fn is_safe_checkpoint_follower(b: u8) -> bool {
    b.is_ascii_graphic() && b != b'\'' && b != b'/'
}

#[inline]
fn hash_prompt_key(tokenizer_id: u64, add_special_tokens: bool, bytes: &[u8]) -> u64 {
    let mut h = tokenizer_id
        ^ ((add_special_tokens as u64) << 63)
        ^ ((bytes.len() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut i = 0usize;
    while i + 8 <= bytes.len() {
        let word = u64::from_le_bytes([
            bytes[i],
            bytes[i + 1],
            bytes[i + 2],
            bytes[i + 3],
            bytes[i + 4],
            bytes[i + 5],
            bytes[i + 6],
            bytes[i + 7],
        ]);
        h = (h ^ word).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        i += 8;
    }
    while i < bytes.len() {
        h = (h ^ (bytes[i] as u64)).wrapping_mul(0x94D0_49BB_1331_11EB);
        i += 1;
    }
    h ^ (h >> 31)
}
