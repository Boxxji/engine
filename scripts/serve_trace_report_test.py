#!/usr/bin/env python3
"""Unit tests for serve_trace_report.py."""

import unittest

from serve_trace_report import (
    aggregate_records,
    check_correctness_and_integrity,
    detect_regressions,
    main,
    parse_log_lines,
    parse_log_lines_with_anomalies,
)


SAMPLE_LOG = """
2026-11-15T09:07:38Z INFO hanzo_engine::engine::ledger: serve_api: batch 1 tok in 0.36s = n/a | prompt 16, prefill 0.36s (16 new)
2026-11-15T09:07:38Z INFO hanzo_engine::engine::ledger: serve_trace: mode=dense bs=none spec_rounds=0 spec_committed=0 plain_steps=0 drafted=0 accepted=0 spec_accept=0.0% spec_commit=0.00/round cuda_decode_graph=capture:0,replay:0 cuda_prefill_graph=capture:0,replay:0 prefill_ms=356.0 decode_ms=357.0 stand_down=none
2026-11-15T09:07:43Z INFO hanzo_engine::engine::ledger: serve_api: dflash 96 tok in 2.21s = 43.44 t/s | 86 rounds, commit 1.10/round | prompt 5020, prefill 2.37s = 2116 t/s
2026-11-15T09:07:43Z INFO hanzo_engine::engine::ledger: serve_trace: mode=dflash->dense bs=8 spec_rounds=4 spec_committed=13 plain_steps=82 drafted=28 accepted=9 spec_accept=32.1% spec_commit=3.25/round cuda_decode_graph=capture:1,replay:77 cuda_prefill_graph=capture:0,replay:0 prefill_ms=2372.0 decode_ms=2214.0 stand_down=slower_than_plain_probe(spec=15.98ms/tok>=est_plain=15.31ms/tok,mean_commit=3.25)
2026-11-15T09:10:22Z INFO hanzo_engine::engine::ledger: serve_api: dflash 96 tok in 2.37s = 40.47 t/s | 90 rounds, commit 1.06/round | prompt 29691, prefill 25.00s = 1187 t/s
2026-11-15T09:10:22Z INFO hanzo_engine::engine::ledger: serve_trace: mode=dflash+lsa->lsa bs=8 spec_rounds=4 spec_committed=9 plain_steps=86 drafted=28 accepted=5 spec_accept=17.9% spec_commit=2.25/round cuda_decode_graph=capture:0,replay:0 cuda_prefill_graph=capture:0,replay:0 prefill_ms=25005.0 decode_ms=2372.0 stand_down=slower_than_plain_probe(spec=35.90ms/tok>=est_plain=16.20ms/tok,mean_commit=2.25)
""".strip().splitlines()


class ServeTraceReportTest(unittest.TestCase):
    def test_parses_pairs_and_filters_short_probe(self):
        recs = parse_log_lines(SAMPLE_LOG, source="sample.log")
        self.assertEqual(len(recs), 3)
        self.assertEqual(recs[0].completion_tokens, 1)
        self.assertEqual(recs[1].prompt_tokens, 5020)
        self.assertEqual(recs[1].cuda_decode_graph_replay, 77)

        report = aggregate_records(
            recs,
            min_completion_tokens=8,
            correctness_payload=[{"name": "case1", "passed": True}],
        )
        tp = report["throughput"]
        self.assertEqual(tp["measured_requests"], 2)
        self.assertEqual(tp["excluded_probe_requests"], 1)

        d_dense = tp["by_mode"]["dflash->dense"]
        self.assertEqual(d_dense["cuda_decode_graph_captures"], 1)
        self.assertEqual(d_dense["cuda_decode_graph_replays"], 77)
        self.assertAlmostEqual(d_dense["cuda_decode_graph_replay_rate_pct"], 93.9, places=1)
        self.assertAlmostEqual(d_dense["spec_accept_rate_pct"], 32.14, places=2)
        self.assertEqual(d_dense["stand_down_rate_pct"], 100.0)
        self.assertEqual(d_dense["stand_down_reasons"], {"slower_than_plain_probe": 1})

        corr = report["correctness"]
        self.assertTrue(corr["trace_invariants_ok"])
        self.assertTrue(corr["external_correctness"]["all_passed"])

    def test_regression_detection(self):
        base = {
            "dense": {
                "ttft_p50_ms": 2000.0,
                "ttft_p95_ms": 2200.0,
                "decode_tok_s_p50": 60.0,
                "total_latency_p50_ms": 3600.0,
                "total_latency_p95_ms": 3800.0,
            }
        }
        cur = {
            "dense": {
                "ttft_p50_ms": 2500.0,
                "ttft_p95_ms": 2250.0,
                "decode_tok_s_p50": 50.0,
                "total_latency_p50_ms": 3700.0,
                "total_latency_p95_ms": 3850.0,
            }
        }
        regs = detect_regressions(cur, base, 15.0, 10.0, 15.0)
        self.assertEqual(len(regs), 2)
        self.assertIn("TTFT p50", regs[0])
        self.assertIn("decode_tok_s p50", regs[1])


API = "INFO ledger: serve_api: dflash {n} tok in 1.00s = 50.00 t/s | 5 rounds, commit 3.00/round | prompt 5000, prefill 2.00s = 2500 t/s seq={seq}"
TRACE = (
    "INFO ledger: serve_trace: seq={seq} mode={mode} bs=8 spec_rounds={rounds} spec_committed={committed} "
    "plain_steps={plain} drafted={drafted} accepted={accepted} spec_accept=0.0% spec_commit=0.00/round "
    "cuda_decode_graph=capture:0,replay:0 cuda_prefill_graph=capture:0,replay:0 "
    "prefill_ms=2000.0 decode_ms=1000.0 stand_down={sd}"
)


def api(n, seq=1):
    return API.format(n=n, seq=seq)


def trace(mode="dflash", rounds=0, committed=0, plain=0, drafted=0, accepted=0, sd="none", seq=1):
    return TRACE.format(
        mode=mode, rounds=rounds, committed=committed, plain=plain,
        drafted=drafted, accepted=accepted, sd=sd, seq=seq,
    )


def anomalies_for(lines):
    recs, pairing = parse_log_lines_with_anomalies(lines, source="t.log")
    return check_correctness_and_integrity(recs, None, pairing)["trace_invariant_anomalies"]


class InvariantTest(unittest.TestCase):
    def test_full_acceptance_round_finishing_request_is_exact(self):
        # 1 prefill token + 1 bootstrap anchor + 2 rounds x (7 accepted + 1 continuation) = 18.
        lines = [api(18), trace(rounds=2, committed=16, plain=1, drafted=14, accepted=14)]
        self.assertEqual(anomalies_for(lines), [])

    def test_partial_acceptance_and_early_eos(self):
        # Round 1: 2 accepted + correction (3). Round 2: EOS on the 1st accepted draft (1).
        lines = [api(6), trace(rounds=2, committed=4, plain=1, drafted=14, accepted=3)]
        self.assertEqual(anomalies_for(lines), [])

    def test_rejection_then_stand_down_plain_decode(self):
        # 1 prefill + 1 anchor + 1 round rejecting every draft (correction only) + 10 plain.
        lines = [
            api(13),
            trace(mode="dflash->dense", rounds=1, committed=1, plain=11, drafted=7, accepted=0,
                  sd="slower_than_plain_probe"),
        ]
        self.assertEqual(anomalies_for(lines), [])

    def test_old_final_round_undercount_fails(self):
        # Pattern from the pre-fix 27B combo log: 96 tokens but 1 + 1 + 90 = 92 counted.
        lines = [api(96), trace(mode="dflash+lsa", rounds=24, committed=90, plain=1,
                                          drafted=175, accepted=70)]
        out = anomalies_for(lines)
        self.assertEqual(len(out), 1)
        self.assertIn("completion_tokens=96", out[0])

    def test_off_by_one_is_not_tolerated(self):
        lines = [api(20), trace(rounds=2, committed=16, plain=1, drafted=14, accepted=14)]
        self.assertEqual(len(anomalies_for(lines)), 1)

    def test_missing_and_orphan_traces_fail(self):
        trailing = [api(1, seq=1), trace(seq=1), api(1, seq=2)]
        out = anomalies_for(trailing)
        self.assertEqual(len(out), 1)
        self.assertIn("serve_api seq=2 has no serve_trace", out[0])
        orphan = [trace(seq=9)]
        self.assertIn("serve_trace seq=9 has no serve_api", anomalies_for(orphan)[0])

    def test_concurrent_interleaved_requests_pair_by_seq(self):
        # Two requests finish interleaved: adjacency pairing would cross them and fail both.
        lines = [
            api(18, seq=3),
            api(13, seq=4),
            trace(mode="dflash->dense", rounds=1, committed=1, plain=11, drafted=7, sd="slower_than_plain_probe", seq=4),
            trace(rounds=2, committed=16, plain=1, drafted=14, accepted=14, seq=3),
        ]
        self.assertEqual(anomalies_for(lines), [])
        recs, _ = parse_log_lines_with_anomalies(lines, source="t.log")
        self.assertEqual([r.completion_tokens for r in recs], [13, 18])

    def test_reused_seq_before_trace_fails(self):
        # A lost trace (cancellation, crash) followed by a restarted process reusing the id.
        lines = [api(5, seq=1), api(1, seq=1), trace(seq=1)]
        out = anomalies_for(lines)
        self.assertEqual(len(out), 1)
        self.assertIn("reused seq=1", out[0])

    def test_legacy_order_pairing_is_flagged_unless_allowed(self):
        legacy_api = "INFO ledger: serve_api: batch 1 tok in 0.10s = n/a | prompt 16, prefill 0.10s (16 new)"
        legacy_trace = TRACE.replace("seq={seq} ", "").format(
            mode="dense", rounds=0, committed=0, plain=0, drafted=0, accepted=0, sd="none")
        recs, out = parse_log_lines_with_anomalies([legacy_api, legacy_trace], source="old.log")
        self.assertEqual(len(recs), 1)
        self.assertTrue(any("paired by log order" in a for a in out))
        _, ok = parse_log_lines_with_anomalies(
            [legacy_api, legacy_trace], source="old.log", allow_legacy_pairing=True)
        self.assertEqual(ok, [])

    def test_zero_token_completion(self):
        api0 = "INFO ledger: serve_api: batch 0 tok in 0.00s = n/a | prompt 10, prefill 0.10s (10 new) seq=1"
        self.assertEqual(anomalies_for([api0, trace(mode="dense")]), [])

    def test_composite_run_and_seq_disambiguates_workers_and_restarts(self):
        lines = [
            "INFO ledger: serve_api: batch 12 tok in 0.40s = 30.00 t/s | prompt 100, prefill 0.10s (100 new) run=a1-0001 seq=0",
            "INFO ledger: serve_api: batch 20 tok in 0.60s = 33.33 t/s | prompt 100, prefill 0.10s (100 new) run=b2-0002 seq=0",
            "INFO ledger: serve_trace: run=b2-0002 seq=0 mode=dense bs=none spec_rounds=0 spec_committed=0 plain_steps=19 drafted=0 accepted=0 spec_accept=0.0% spec_commit=0.00/round cuda_decode_graph=capture:0,replay:19 cuda_prefill_graph=capture:0,replay:0 prefill_ms=100.0 decode_ms=600.0 stand_down=none status=completed",
            "INFO ledger: serve_trace: run=a1-0001 seq=0 mode=dense bs=none spec_rounds=0 spec_committed=0 plain_steps=11 drafted=0 accepted=0 spec_accept=0.0% spec_commit=0.00/round cuda_decode_graph=capture:0,replay:11 cuda_prefill_graph=capture:0,replay:0 prefill_ms=100.0 decode_ms=400.0 stand_down=none status=completed",
        ]
        recs, pairing = parse_log_lines_with_anomalies(lines, source="multi.log")
        self.assertEqual(pairing, [])
        report = aggregate_records(recs, min_completion_tokens=8, pairing_anomalies=pairing)
        self.assertTrue(report["correctness"]["trace_invariants_ok"])
        self.assertEqual([r.completion_tokens for r in recs], [20, 12])

    def test_canceled_and_error_excluded_from_throughput(self):
        lines = [
            "INFO ledger: serve_api: batch 16 tok in 0.40s = 40.00 t/s | prompt 100, prefill 0.10s (100 new) run=a1 seq=1",
            "INFO ledger: serve_trace: run=a1 seq=1 mode=dense bs=none spec_rounds=0 spec_committed=0 plain_steps=15 drafted=0 accepted=0 spec_accept=0.0% spec_commit=0.00/round cuda_decode_graph=capture:1,replay:14 cuda_prefill_graph=capture:0,replay:0 prefill_ms=100.0 decode_ms=400.0 stand_down=none status=completed",
            "INFO ledger: serve_api: batch 4 tok in 0.20s = 20.00 t/s | prompt 100, prefill 0.10s (100 new) run=a1 seq=2",
            "INFO ledger: serve_trace: run=a1 seq=2 mode=dense bs=none spec_rounds=0 spec_committed=0 plain_steps=4 drafted=0 accepted=0 spec_accept=0.0% spec_commit=0.00/round cuda_decode_graph=capture:0,replay:4 cuda_prefill_graph=capture:0,replay:0 prefill_ms=100.0 decode_ms=200.0 stand_down=none status=canceled",
            "INFO ledger: serve_api: batch 0 tok in 0.00s = n/a | prompt 100, prefill 0.05s (100 new) run=a1 seq=3",
            "INFO ledger: serve_trace: run=a1 seq=3 mode=dense bs=none spec_rounds=0 spec_committed=0 plain_steps=0 drafted=0 accepted=0 spec_accept=0.0% spec_commit=0.00/round cuda_decode_graph=capture:0,replay:0 cuda_prefill_graph=capture:0,replay:0 prefill_ms=50.0 decode_ms=0.0 stand_down=none status=error",
        ]
        recs, pairing = parse_log_lines_with_anomalies(lines, source="lc.log")
        report = aggregate_records(recs, min_completion_tokens=4, pairing_anomalies=pairing)
        self.assertTrue(report["correctness"]["trace_invariants_ok"])
        self.assertEqual(report["lifecycle"]["completed"], 1)
        self.assertEqual(report["lifecycle"]["canceled"], 1)
        self.assertEqual(report["lifecycle"]["error"], 1)
        self.assertEqual(report["throughput"]["measured_requests"], 1)
        self.assertEqual(report["throughput"]["excluded_non_completed_requests"], 2)

    def test_main_exits_nonzero_on_inconsistent_trace(self):
        import os
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.write(api(96) + "\n")
            f.write(trace(mode="dflash+lsa", rounds=24, committed=90, plain=1, drafted=175,
                          accepted=70) + "\n")
            path = f.name
        try:
            self.assertEqual(main([path]), 3)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
