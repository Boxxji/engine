//! Incremental suffix automaton for `O(ngram_max)` prompt-lookup and `SA_SPEC` matching.
//!
//! Tracks `toks[..toks.len() - 1]` with `O(1)` amortized append and rollback so
//! normal autoregressive token appends and speculative draft rollbacks avoid
//! rebuilding from scratch. For `best_n >= 3` on internal SAM states, exact
//! most-recent occurrence lookup intersects backward 2-gram chains from both the
//! tail and head of the matched needle in lockstep, preserving 100% parity with
//! `super::prompt_lookup` matching.

use ahash::AHashMap;

const LINK_NONE: u32 = u32::MAX;

#[inline]
fn pack2(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | (b as u64)
}

#[derive(Clone, Default, Debug)]
struct SmallEdges {
    inline: [(u32, u32); 4],
    len: u8,
    overflow: Option<Box<AHashMap<u32, u32>>>,
}

impl SmallEdges {
    #[inline]
    fn get(&self, key: u32) -> Option<u32> {
        for &(k, v) in &self.inline[..self.len as usize] {
            if k == key {
                return Some(v);
            }
        }
        self.overflow.as_deref().and_then(|m| m.get(&key).copied())
    }

    #[inline]
    fn insert(&mut self, key: u32, val: u32) -> Option<u32> {
        for entry in &mut self.inline[..self.len as usize] {
            if entry.0 == key {
                let prev = entry.1;
                entry.1 = val;
                return Some(prev);
            }
        }
        if let Some(map) = self.overflow.as_deref_mut() {
            if let Some(prev) = map.get_mut(&key) {
                let old = *prev;
                *prev = val;
                return Some(old);
            }
        }
        if (self.len as usize) < self.inline.len() {
            self.inline[self.len as usize] = (key, val);
            self.len += 1;
            return None;
        }
        let map = self
            .overflow
            .get_or_insert_with(|| Box::new(AHashMap::new()));
        map.insert(key, val)
    }

    #[inline]
    fn remove(&mut self, key: u32) {
        let len = self.len as usize;
        for i in 0..len {
            if self.inline[i].0 == key {
                self.len -= 1;
                self.inline[i] = self.inline[self.len as usize];
                return;
            }
        }
        if let Some(map) = self.overflow.as_deref_mut() {
            map.remove(&key);
        }
    }
}

#[derive(Clone, Debug)]
struct State {
    len: u32,
    link: u32,
    /// A valid 1-based exclusive end position in `prefix` for strings in this class.
    last_end: u32,
    /// Number of states whose suffix `link` points directly to this state.
    child_count: u32,
    edges: SmallEdges,
}

impl Default for State {
    fn default() -> Self {
        Self {
            len: 0,
            link: LINK_NONE,
            last_end: 0,
            child_count: 0,
            edges: SmallEdges::default(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Checkpoint {
    states_len: usize,
    prev_last: u32,
    edge_undo_len: usize,
    link_undo_len: usize,
    prev_1gram: Option<u32>,
    prev_2gram_occ: u32,
}

/// Incremental suffix automaton over `u32` token sequences supporting fast append
/// and speculative-rejection rollback.
#[derive(Clone, Debug)]
pub struct SuffixAutomaton {
    states: Vec<State>,
    last: u32,
    prefix: Vec<u32>,
    last_1gram: AHashMap<u32, u32>,
    last_2gram: AHashMap<u64, u32>,
    /// 1-based exclusive end position chain for 2-grams (`prev_2gram[end_excl]`).
    prev_2gram: Vec<u32>,
    edge_undo: Vec<(u32, u32, Option<u32>)>,
    link_undo: Vec<(u32, u32)>,
    checkpoints: Vec<Checkpoint>,
}

impl Default for SuffixAutomaton {
    fn default() -> Self {
        Self::new()
    }
}

impl SuffixAutomaton {
    /// Construct an empty suffix automaton.
    pub fn new() -> Self {
        Self {
            states: vec![State::default()],
            last: 0,
            prefix: Vec::new(),
            last_1gram: AHashMap::new(),
            last_2gram: AHashMap::new(),
            prev_2gram: vec![0],
            edge_undo: Vec::new(),
            link_undo: Vec::new(),
            checkpoints: Vec::new(),
        }
    }

    /// Clear all automaton state while retaining allocated capacity.
    pub fn clear(&mut self) {
        self.states.truncate(1);
        self.states[0] = State::default();
        self.last = 0;
        self.prefix.clear();
        self.last_1gram.clear();
        self.last_2gram.clear();
        self.prev_2gram.clear();
        self.prev_2gram.push(0);
        self.edge_undo.clear();
        self.link_undo.clear();
        self.checkpoints.clear();
    }

    #[inline]
    fn set_link(&mut self, state: u32, new_link: u32) {
        let old_link = self.states[state as usize].link;
        if old_link == new_link {
            return;
        }
        self.link_undo.push((state, old_link));
        if old_link != LINK_NONE {
            self.states[old_link as usize].child_count -= 1;
        }
        if new_link != LINK_NONE {
            self.states[new_link as usize].child_count += 1;
        }
        self.states[state as usize].link = new_link;
    }

    #[inline]
    fn set_edge(&mut self, state: u32, tok: u32, target: u32) {
        let prev = self.states[state as usize].edges.insert(tok, target);
        self.edge_undo.push((state, tok, prev));
    }

    /// Append a single token to the indexed prefix in `O(1)` amortized time.
    pub fn extend_one(&mut self, tok: u32) {
        let edge_undo_len = self.edge_undo.len();
        let link_undo_len = self.link_undo.len();
        let states_len = self.states.len();
        let prev_last = self.last;

        self.prefix.push(tok);
        let end_excl = u32::try_from(self.prefix.len()).expect("SAM prefix length exceeds u32");

        let prev_1gram = self.last_1gram.insert(tok, end_excl);
        let prev_2gram_occ = if end_excl >= 2 {
            let prev_tok = self.prefix[self.prefix.len() - 2];
            self.last_2gram
                .insert(pack2(prev_tok, tok), end_excl)
                .unwrap_or(0)
        } else {
            0
        };
        self.prev_2gram.push(prev_2gram_occ);

        let cur = u32::try_from(self.states.len()).expect("SAM state count exceeds u32");
        self.states.push(State {
            len: self.states[prev_last as usize].len + 1,
            link: LINK_NONE,
            last_end: end_excl,
            child_count: 0,
            edges: SmallEdges::default(),
        });

        let mut p = prev_last;
        while p != LINK_NONE && self.states[p as usize].edges.get(tok).is_none() {
            self.set_edge(p, tok, cur);
            p = self.states[p as usize].link;
        }

        if p == LINK_NONE {
            self.set_link(cur, 0);
        } else {
            let q = self.states[p as usize].edges.get(tok).unwrap();
            if self.states[p as usize].len + 1 == self.states[q as usize].len {
                self.set_link(cur, q);
            } else {
                let clone = u32::try_from(self.states.len()).expect("SAM state count exceeds u32");
                let q_state = &self.states[q as usize];
                let cloned = State {
                    len: self.states[p as usize].len + 1,
                    link: LINK_NONE,
                    last_end: q_state.last_end,
                    child_count: 0,
                    edges: q_state.edges.clone(),
                };
                self.states.push(cloned);
                let q_old_link = self.states[q as usize].link;
                self.set_link(clone, q_old_link);

                while p != LINK_NONE && self.states[p as usize].edges.get(tok) == Some(q) {
                    self.set_edge(p, tok, clone);
                    p = self.states[p as usize].link;
                }
                self.set_link(q, clone);
                self.set_link(cur, clone);
            }
        }

        self.last = cur;
        self.checkpoints.push(Checkpoint {
            states_len,
            prev_last,
            edge_undo_len,
            link_undo_len,
            prev_1gram,
            prev_2gram_occ,
        });
    }

    /// Roll back the most recently appended token in `O(1)` amortized time.
    pub fn rollback_one(&mut self) {
        let Some(cp) = self.checkpoints.pop() else {
            return;
        };

        while self.link_undo.len() > cp.link_undo_len {
            let (state, old_link) = self.link_undo.pop().unwrap();
            let cur_link = self.states[state as usize].link;
            if cur_link != LINK_NONE {
                self.states[cur_link as usize].child_count -= 1;
            }
            if old_link != LINK_NONE {
                self.states[old_link as usize].child_count += 1;
            }
            self.states[state as usize].link = old_link;
        }

        while self.edge_undo.len() > cp.edge_undo_len {
            let (state, tok, prev) = self.edge_undo.pop().unwrap();
            match prev {
                Some(target) => {
                    self.states[state as usize].edges.insert(tok, target);
                }
                None => {
                    self.states[state as usize].edges.remove(tok);
                }
            }
        }

        self.states.truncate(cp.states_len);
        self.last = cp.prev_last;

        let tok = self.prefix.pop().unwrap();
        self.prev_2gram.pop();
        match cp.prev_1gram {
            Some(prev_pos) => {
                self.last_1gram.insert(tok, prev_pos);
            }
            None => {
                self.last_1gram.remove(&tok);
            }
        }
        if !self.prefix.is_empty() {
            let prev_tok = *self.prefix.last().unwrap();
            let key2 = pack2(prev_tok, tok);
            if cp.prev_2gram_occ == 0 {
                self.last_2gram.remove(&key2);
            } else {
                self.last_2gram.insert(key2, cp.prev_2gram_occ);
            }
        }
    }

    /// Synchronize the indexed prefix with `target_prefix` incrementally.
    pub fn sync_prefix(&mut self, target_prefix: &[u32]) {
        let common = self
            .prefix
            .iter()
            .zip(target_prefix.iter())
            .take_while(|(a, b)| a == b)
            .count();

        if common == 0 && !self.prefix.is_empty() && target_prefix.len() > 64 {
            self.clear();
        } else {
            while self.prefix.len() > common {
                self.rollback_one();
            }
        }

        let start = self.prefix.len();
        let rem = target_prefix.len() - start;
        if rem > 0 {
            self.states.reserve(rem.saturating_mul(2));
            self.prefix.reserve(rem);
            self.prev_2gram.reserve(rem);
            self.checkpoints.reserve(rem);
            self.edge_undo.reserve(rem.saturating_mul(2));
            self.link_undo.reserve(rem.saturating_mul(2));
            self.last_1gram.reserve(rem.min(4096));
            self.last_2gram.reserve(rem.min(8192));
        }
        for &tok in &target_prefix[start..] {
            self.extend_one(tok);
        }
    }

    /// Find the longest suffix match of `toks` in `toks[..toks.len() - 1]` within
    /// `ngram_min..=ngram_max`, returning `(matched_ngram_len, continuation[..<=gamma])`.
    ///
    /// Returns the continuation after the **most recent** earlier occurrence of
    /// that longest matched needle, matching `super::prompt_lookup` identically.
    pub fn lookup(
        &mut self,
        toks: &[u32],
        ngram_min: usize,
        ngram_max: usize,
        gamma: usize,
    ) -> (usize, Vec<u32>) {
        if gamma == 0 || ngram_min == 0 || ngram_max < ngram_min {
            return (0, Vec::new());
        }
        let len = toks.len();
        if len <= 1 {
            return (0, Vec::new());
        }
        let p_len = len - 1;
        if p_len < ngram_min {
            return (0, Vec::new());
        }

        self.sync_prefix(&toks[..p_len]);

        let longest = ngram_max.min(p_len);
        let window = &toks[len - longest..];
        let mut v = 0usize;
        let mut cur_len = 0usize;
        for &tok in window {
            while v != 0 && self.states[v].edges.get(tok).is_none() {
                v = self.states[v].link as usize;
                cur_len = self.states[v].len as usize;
            }
            if let Some(next) = self.states[v].edges.get(tok) {
                v = next as usize;
                cur_len += 1;
            } else {
                cur_len = 0;
            }
        }

        let best_n = cur_len;
        if best_n < ngram_min {
            return (0, Vec::new());
        }

        let needle = &toks[len - best_n..];
        let end_excl = match best_n {
            1 => self.last_1gram[&needle[0]] as usize,
            2 => self.last_2gram[&pack2(needle[0], needle[1])] as usize,
            _ => {
                let floor = self.states[v].last_end as usize;
                if self.states[v].child_count == 0 {
                    floor
                } else {
                    self.most_recent_match_above(needle, floor)
                }
            }
        };

        let from = end_excl;
        let to = (from + gamma).min(len);
        (best_n, toks[from..to].to_vec())
    }

    fn most_recent_match_above(&self, needle: &[u32], floor: usize) -> usize {
        let best_n = needle.len();
        let p_len = self.prefix.len();
        let tail_key = pack2(needle[best_n - 2], needle[best_n - 1]);
        let head_key = pack2(needle[0], needle[1]);

        let mut tail_pos = self.last_2gram.get(&tail_key).copied().unwrap_or(0) as usize;
        let mut head_pos = self.last_2gram.get(&head_key).copied().unwrap_or(0) as usize;

        loop {
            while head_pos >= 2 && head_pos - 2 + best_n > p_len {
                head_pos = self.prev_2gram[head_pos] as usize;
            }
            let h_done = head_pos < 2 || head_pos - 2 + best_n <= floor;
            if tail_pos <= floor || h_done {
                return floor;
            }

            if tail_pos >= best_n && &self.prefix[tail_pos - best_n..tail_pos] == needle {
                return tail_pos;
            }
            tail_pos = self.prev_2gram[tail_pos] as usize;

            let h_cand = head_pos - 2 + best_n;
            if &self.prefix[h_cand - best_n..h_cand] == needle {
                return h_cand;
            }
            head_pos = self.prev_2gram[head_pos] as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SuffixAutomaton;
    use rand::{Rng, SeedableRng};
    use rand_isaac::Isaac64Rng;

    fn reference_lookup(
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

    #[test]
    fn edge_cases_and_periodic_sequences_match_reference() {
        let mut sa = SuffixAutomaton::new();
        assert_eq!(sa.lookup(&[], 1, 4, 4), (0, Vec::new()));
        assert_eq!(sa.lookup(&[1], 1, 4, 4), (0, Vec::new()));
        assert_eq!(sa.lookup(&[1, 1], 1, 4, 4), (1, vec![1]));

        let all_ones = vec![1u32; 4096];
        assert_eq!(
            sa.lookup(&all_ones, 1, 8, 6),
            reference_lookup(&all_ones, 1, 8, 6)
        );

        let mut alt = Vec::with_capacity(4096);
        for _ in 0..2048 {
            alt.push(1);
            alt.push(2);
        }
        assert_eq!(sa.lookup(&alt, 2, 12, 8), reference_lookup(&alt, 2, 12, 8));

        // Adversarial tail-heavy 2-gram chain: [8, 1, 2] + [9, 1, 2]^1024 + [8, 1, 2]
        let mut tail_heavy = vec![8, 1, 2];
        for _ in 0..1024 {
            tail_heavy.extend_from_slice(&[9, 1, 2]);
        }
        tail_heavy.extend_from_slice(&[8, 1, 2]);
        assert_eq!(
            sa.lookup(&tail_heavy, 2, 5, 4),
            reference_lookup(&tail_heavy, 2, 5, 4)
        );

        // Adversarial head-heavy 2-gram chain: [8, 1, 2] + [8, 1, 9]^1024 + [8, 1, 2]
        let mut head_heavy = vec![8, 1, 2];
        for _ in 0..1024 {
            head_heavy.extend_from_slice(&[8, 1, 9]);
        }
        head_heavy.extend_from_slice(&[8, 1, 2]);
        assert_eq!(
            sa.lookup(&head_heavy, 2, 5, 4),
            reference_lookup(&head_heavy, 2, 5, 4)
        );
    }

    #[test]
    fn differential_fuzz_against_reference_across_appends_and_rollbacks() {
        let mut rng = Isaac64Rng::seed_from_u64(0x5A_5F_EC_12_34_56_78_9A);

        for &vocab in &[2u32, 4, 16, 512] {
            let mut sa = SuffixAutomaton::new();
            let mut toks = Vec::new();
            for _ in 0..500 {
                let op = rng.random_range(0..10);
                if op < 7 || toks.is_empty() {
                    let count = rng.random_range(1..=6);
                    for _ in 0..count {
                        toks.push(rng.random_range(0..vocab));
                    }
                } else if op < 9 {
                    let drop_n = rng.random_range(1..=toks.len().min(8));
                    toks.truncate(toks.len() - drop_n);
                } else {
                    // Replace a short suffix to test rollback + re-append path.
                    let drop_n = rng.random_range(1..=toks.len().min(5));
                    toks.truncate(toks.len() - drop_n);
                    let add_n = rng.random_range(1..=5);
                    for _ in 0..add_n {
                        toks.push(rng.random_range(0..vocab));
                    }
                }

                let ngram_min = rng.random_range(1..=3);
                let ngram_max = rng.random_range(ngram_min..=8);
                let gamma = rng.random_range(1..=6);

                let got = sa.lookup(&toks, ngram_min, ngram_max, gamma);
                let expected = reference_lookup(&toks, ngram_min, ngram_max, gamma);
                assert_eq!(
                    got, expected,
                    "mismatch for vocab={vocab}, len={}, min={ngram_min}, max={ngram_max}, gamma={gamma}",
                    toks.len()
                );
            }
        }
    }
}
