//! Per-sequence adaptive speculative regression guard.
//!
//! Tracks rolling verified acceptance per proposer path and temporarily cools
//! down speculation when acceptance drops below breakeven, then probes one step
//! after cooldown so shifting from chat back to code edits recovers automatically.

use ahash::AHashMap;

#[derive(Clone, Copy, Debug, Default)]
struct SeqGuardState {
    prev_proposed: usize,
    prev_accepted: usize,
    armed: bool,
    has_sample: bool,
    ewma_accepted: f32,
    low_streak: u32,
    backoff_shift: u32,
    cooldown_left: u32,
}

const MAX_BACKOFF_SHIFT: u32 = 2;
const LATENCY_BATCH_MAX: usize = 4;

/// Bounded cooldown regression guard keyed by `seq_id`.
#[derive(Clone, Debug)]
pub struct AdaptiveSpeculativeGuard {
    min_accepted: f32,
    max_low_streak: u32,
    cooldown_steps: u32,
    seqs: AHashMap<usize, SeqGuardState>,
    carry_across_seqs: bool,
    carried_backoff_shift: u32,
    carried_cooldown_left: u32,
}

impl AdaptiveSpeculativeGuard {
    pub fn new(min_accepted: f32, max_low_streak: u32, cooldown_steps: u32) -> Self {
        Self {
            min_accepted,
            max_low_streak: max_low_streak.max(1),
            cooldown_steps: cooldown_steps.max(1),
            seqs: AHashMap::new(),
            carry_across_seqs: false,
            carried_backoff_shift: 0,
            carried_cooldown_left: 0,
        }
    }

    /// Share backoff and remaining cooldown across sequential requests (e.g. multi-turn
    /// conversations where each turn receives a fresh sequence id).
    pub fn with_cross_seq_carry(mut self, carry: bool) -> Self {
        self.carry_across_seqs = carry;
        self
    }

    fn ensure_seq(&mut self, seq_id: usize) -> &mut SeqGuardState {
        if !self.seqs.contains_key(&seq_id) {
            let mut st = SeqGuardState::default();
            if self.carry_across_seqs {
                st.backoff_shift = self.carried_backoff_shift;
                st.cooldown_left = self.carried_cooldown_left;
            }
            self.seqs.insert(seq_id, st);
        }
        self.seqs.get_mut(&seq_id).expect("inserted above")
    }

    fn sync_carried(&mut self) {
        if !self.carry_across_seqs || self.seqs.is_empty() {
            return;
        }
        self.carried_cooldown_left = self
            .seqs
            .values()
            .map(|s| s.cooldown_left)
            .min()
            .unwrap_or(0);
        self.carried_backoff_shift = self
            .seqs
            .values()
            .map(|s| s.backoff_shift)
            .min()
            .unwrap_or(0);
    }

    /// Observe `current_counts = (total_draft_toks, total_draft_accepted)` for `seq_id` and
    /// update EWMA acceptance / cooldown state without ticking an active cooldown counter.
    /// Calling this repeatedly in the same round before `mark_proposed` is idempotent.
    pub fn observe(&mut self, seq_id: usize, current_counts: (usize, usize)) {
        let min_accepted = self.min_accepted;
        let max_low_streak = self.max_low_streak;
        let cooldown_steps = self.cooldown_steps;
        let st = self.ensure_seq(seq_id);
        let delta_prop = current_counts.0.saturating_sub(st.prev_proposed);
        let delta_acc = current_counts.1.saturating_sub(st.prev_accepted);
        st.prev_proposed = current_counts.0;
        st.prev_accepted = current_counts.1;

        if st.armed && delta_prop > 0 {
            let acc_f = f32::from(u16::try_from(delta_acc).unwrap_or(u16::MAX));
            if !st.has_sample {
                st.ewma_accepted = acc_f;
                st.has_sample = true;
            } else {
                st.ewma_accepted = 0.5 * st.ewma_accepted + 0.5 * acc_f;
            }
            if acc_f < min_accepted && st.ewma_accepted < min_accepted {
                st.low_streak = st.low_streak.saturating_add(1);
            } else {
                st.low_streak = 0;
                st.backoff_shift = 0;
            }
            let trigger_streak = if st.backoff_shift > 0 {
                1
            } else {
                max_low_streak
            };
            if st.low_streak >= trigger_streak {
                let shift = st.backoff_shift.min(MAX_BACKOFF_SHIFT);
                st.cooldown_left = cooldown_steps.saturating_mul(1u32 << shift);
                st.backoff_shift = st.backoff_shift.saturating_add(1).min(MAX_BACKOFF_SHIFT);
                st.low_streak = 0;
            }
        }
        st.armed = false;
        self.sync_carried();
    }

    /// Return `true` if `seq_id` is allowed to speculate this step; otherwise consume one
    /// cooldown step, record a guard skip, and return `false`.
    pub fn check_or_tick_cooldown(&mut self, seq_id: usize) -> bool {
        let st = self.ensure_seq(seq_id);
        let allowed = if st.cooldown_left > 0 {
            st.cooldown_left -= 1;
            crate::speculative::stats::record_guard_skip();
            false
        } else {
            true
        };
        self.sync_carried();
        allowed
    }

    /// Whether `seq_id` still has cooldown steps remaining before its next probe.
    pub fn in_cooldown(&self, seq_id: usize) -> bool {
        self.seqs
            .get(&seq_id)
            .map(|s| s.cooldown_left > 0)
            .unwrap_or(self.carry_across_seqs && self.carried_cooldown_left > 0)
    }

    /// Whether every live sequence (or carried session state when idle) is currently in
    /// cooldown, allowing target forwards to skip hidden capture and keep decode graphs enabled.
    pub fn global_in_cooldown(&self) -> bool {
        if self.seqs.is_empty() {
            self.carry_across_seqs && self.carried_cooldown_left > 0
        } else {
            self.seqs.values().all(|s| s.cooldown_left > 0)
        }
    }

    /// Observe `current_counts = (total_draft_toks, total_draft_accepted)` for
    /// `seq_id` and return whether the guarded proposer path should run this step.
    pub fn allow(&mut self, seq_id: usize, current_counts: (usize, usize)) -> bool {
        self.observe(seq_id, current_counts);
        self.check_or_tick_cooldown(seq_id)
    }

    /// Arm outcome tracking for `seq_id` when the guarded path actually emits
    /// speculative tokens on this step.
    pub fn mark_proposed(&mut self, seq_id: usize) {
        let st = self.ensure_seq(seq_id);
        st.armed = true;
    }

    /// Drop state for sequences no longer active in the scheduler.
    pub fn retain_seqs(&mut self, live: &[usize]) {
        self.sync_carried();
        self.seqs.retain(|id, _| live.contains(id));
    }
}

/// Self-guarding wrapper around a neural speculative proposer (e.g. an MTP head).
///
/// Tracks accepted draft tokens via [`AdaptiveSpeculativeGuard`] and stands down during
/// exponentially backed-off cooldown windows when acceptance falls below threshold. While in
/// cooldown, `SpeculativeProposer::needs_target_hiddens` returns `false` so target decode
/// forwards skip hidden capture and re-enter CUDA/ROCm/Vulkan decode graphs.
pub struct GuardedDraft {
    inner: Box<dyn super::proposer::SpeculativeProposer + Send + Sync>,
    guard: AdaptiveSpeculativeGuard,
}

impl GuardedDraft {
    /// Wrap `inner` with the default neural MTP guard: require EWMA >= `1.2` accepted tokens
    /// per round, enter `16`-step cooldown after `2` low-yield rounds, and carry cooldown policy
    /// across sequential turns.
    pub fn new(inner: Box<dyn super::proposer::SpeculativeProposer + Send + Sync>) -> Self {
        Self::with_guard(
            inner,
            AdaptiveSpeculativeGuard::new(1.2, 2, 16).with_cross_seq_carry(true),
        )
    }

    /// Wrap `inner` with a caller-supplied [`AdaptiveSpeculativeGuard`].
    pub fn with_guard(
        inner: Box<dyn super::proposer::SpeculativeProposer + Send + Sync>,
        guard: AdaptiveSpeculativeGuard,
    ) -> Self {
        Self { inner, guard }
    }

    pub fn in_cooldown(&self) -> bool {
        self.guard.global_in_cooldown()
    }
}

impl super::proposer::SpeculativeProposer for GuardedDraft {
    fn proposal_len(&self) -> usize {
        self.inner.proposal_len()
    }

    fn needs_target_hiddens(&self) -> bool {
        !self.guard.global_in_cooldown() && self.inner.needs_target_hiddens()
    }

    fn observe_outcomes(&mut self, seq_ids: &[usize], sequences: &[&crate::sequence::Sequence]) {
        for (&seq_id, seq) in seq_ids.iter().zip(sequences.iter()) {
            self.guard.observe(seq_id, seq.draft_counts());
        }
        self.inner.observe_outcomes(seq_ids, sequences);
    }

    fn propose(
        &mut self,
        ctx: super::proposer::SpeculativeProposeBatchCtx<'_>,
        target_embedder: Option<&super::proposer::TargetTokenEmbedder<'_>>,
    ) -> hanzo_ml::Result<super::proposer::SpeculativeProposalBatch> {
        self.observe_outcomes(ctx.seq_ids, ctx.sequences);
        if ctx.sampled_tokens.len() > LATENCY_BATCH_MAX {
            return Ok(super::proposer::SpeculativeProposalBatch::new(vec![
                super::proposer::SpeculativeProposal::new(Vec::new());
                ctx.sampled_tokens.len()
            ]));
        }
        let mut cooled = false;
        for (&seq_id, seq) in ctx.seq_ids.iter().zip(ctx.sequences.iter()) {
            if !self.guard.check_or_tick_cooldown(seq_id) {
                seq.record_speculative_stand_down("adaptive_guard_cooldown");
                cooled = true;
            }
        }
        if cooled {
            return Ok(super::proposer::SpeculativeProposalBatch::new(vec![
                super::proposer::SpeculativeProposal::new(Vec::new());
                ctx.sampled_tokens.len()
            ]));
        }
        if self.inner.needs_target_hiddens()
            && ctx.target_hiddens.is_none()
            && ctx.target_hidden_layers.is_none()
        {
            return Ok(super::proposer::SpeculativeProposalBatch::new(vec![
                super::proposer::SpeculativeProposal::new(Vec::new());
                ctx.sampled_tokens.len()
            ]));
        }
        let seq_ids = ctx.seq_ids.to_vec();
        let batch = self.inner.propose(ctx, target_embedder)?;
        for (seq_id, proposal) in seq_ids.into_iter().zip(&batch.proposals) {
            if !proposal.is_empty() {
                self.guard.mark_proposed(seq_id);
            }
        }
        Ok(batch)
    }

    fn retain_seqs(&mut self, live: &[usize]) {
        self.guard.retain_seqs(live);
        self.inner.retain_seqs(live);
    }

    fn is_paused_for_seq(&self, seq_id: Option<usize>) -> bool {
        let guard_paused = seq_id.map_or_else(
            || self.guard.global_in_cooldown(),
            |id| self.guard.in_cooldown(id),
        );
        guard_paused || self.inner.is_paused_for_seq(seq_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::test_sequence;
    use crate::speculative::proposer::{
        SpeculativeProposal, SpeculativeProposalBatch, SpeculativeProposeBatchCtx,
        SpeculativeProposer, TargetTokenEmbedder,
    };
    use crate::speculative::SpeculativeKvCache;
    use rand::SeedableRng;
    use rand_isaac::Isaac64Rng;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    };

    struct CountingMtp {
        calls: Arc<AtomicUsize>,
    }

    impl SpeculativeProposer for CountingMtp {
        fn proposal_len(&self) -> usize {
            2
        }

        fn propose(
            &mut self,
            ctx: SpeculativeProposeBatchCtx<'_>,
            _target_embedder: Option<&TargetTokenEmbedder<'_>>,
        ) -> hanzo_ml::Result<SpeculativeProposalBatch> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(SpeculativeProposalBatch::new(
                ctx.seq_ids
                    .iter()
                    .map(|_| SpeculativeProposal::new(vec![10, 11]))
                    .collect(),
            ))
        }
    }

    #[test]
    fn guarded_draft_cools_down_and_reenables_decode_graphs() -> hanzo_ml::Result<()> {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut draft = GuardedDraft::with_guard(
            Box::new(CountingMtp {
                calls: Arc::clone(&calls),
            }),
            AdaptiveSpeculativeGuard::new(1.2, 2, 4).with_cross_seq_carry(true),
        );

        let seq = test_sequence(vec![1, 2, 3, 4], None);
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(0)));
        let dummy_hidden =
            hanzo_ml::Tensor::zeros((1, 4), hanzo_ml::DType::F32, &hanzo_ml::Device::Cpu)?;
        let run_with_hidden =
            |draft: &mut GuardedDraft,
             seq: &crate::sequence::Sequence,
             id: usize,
             target_hiddens: Option<hanzo_ml::Tensor>| {
                let seq_ids = [id];
                let sampled = [*seq.get_toks().last().unwrap()];
                let base_lens = [seq.get_toks().len() - 1];
                let seqs = [seq];
                let ctx = SpeculativeProposeBatchCtx {
                    sampled_tokens: &sampled,
                    sampled_tokens_emitted: true,
                    seq_ids: &seq_ids,
                    base_lens: &base_lens,
                    sequences: &seqs,
                    cache: SpeculativeKvCache::Normal,
                    target_hiddens,
                    target_hidden_layers: None,
                    rng: Arc::clone(&rng),
                };
                draft.propose(ctx, None)
            };
        let run_round = |draft: &mut GuardedDraft, seq: &crate::sequence::Sequence, id: usize| {
            run_with_hidden(draft, seq, id, Some(dummy_hidden.clone()))
        };

        // Missing target hiddens stands down safely without calling the inner MTP head.
        let p_no_hid = run_with_hidden(&mut draft, &seq, 1, None)?;
        assert!(p_no_hid.proposals[0].tokens.is_empty());
        assert_eq!(calls.load(Ordering::Relaxed), 0);

        assert!(draft.needs_target_hiddens());
        let p0 = run_round(&mut draft, &seq, 1)?;
        assert_eq!(p0.proposals[0].tokens, vec![10, 11]);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        // Simulate 0 accepted tokens from round 0.
        seq.record_drafts(2, 0);

        let p1 = run_round(&mut draft, &seq, 1)?;
        assert_eq!(p1.proposals[0].tokens, vec![10, 11]);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        // Second consecutive low-yield round triggers cooldown.
        seq.record_drafts(2, 0);

        // Step 2: enters 4-step cooldown (ticks 4 -> 3), returns empty, re-enables graphs!
        let p2 = run_round(&mut draft, &seq, 1)?;
        assert!(p2.proposals[0].tokens.is_empty());
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        assert!(
            !draft.needs_target_hiddens(),
            "cooldown should suspend target hidden capture so decode graphs re-enable"
        );

        // Steps 3 & 4 stay in cooldown (ticks 3 -> 2 -> 1).
        for _ in 0..2 {
            let p = run_round(&mut draft, &seq, 1)?;
            assert!(p.proposals[0].tokens.is_empty());
            assert!(!draft.needs_target_hiddens());
        }

        // Across turns: if seq 1 finishes mid-cooldown and seq 2 starts, cooldown carries over.
        draft.retain_seqs(&[]);
        assert!(!draft.needs_target_hiddens());

        // Step 5 (final cooldown tick 1 -> 0): skips propose on this step, AND clears
        // global_in_cooldown so the target forward on this step captures hiddens for next step!
        let seq2 = test_sequence(vec![5, 6, 7], None);
        let p5 = run_round(&mut draft, &seq2, 2)?;
        assert!(p5.proposals[0].tokens.is_empty());
        assert!(
            draft.needs_target_hiddens(),
            "final cooldown tick must re-arm hidden capture before the next probe"
        );

        // Step 6: probe runs inner draft again.
        let p6 = run_round(&mut draft, &seq2, 2)?;
        assert_eq!(p6.proposals[0].tokens, vec![10, 11]);
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        Ok(())
    }
}
