#!/usr/bin/env python3
"""Aggregate Hanzo Engine `serve_trace:` / `serve_api:` logs and check regressions.

Pairs each `serve_trace:` record with its immediately preceding `serve_api:` ledger
line (when present) to compute mode-stratified throughput and graph/speculation
telemetry, while keeping output-correctness / trace-integrity checks strictly
separate from throughput metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

SERVE_API_RE = re.compile(
    r"serve_api:\s+(?P<name>\S+)\s+(?P<generated>\d+)\s+tok\s+in\s+(?P<decode_s>[\d.]+)s\s*=\s*(?P<rate>[\d.]+ t/s|n/a)\s*\|\s*"
    r"(?:(?P<rounds>\d+)\s+rounds,\s+commit\s+(?P<commit>[\d.]+)/round\s*\|\s*)?"
    r"prompt\s+(?P<prompt>\d+)"
    r"(?:\s*\((?P<cached>\d+)\s+cached,\s+(?P<cached_pct>[\d.]+)%\))?,\s*"
    r"prefill\s+(?P<prefill_s>[\d.]+)s"
)

SERVE_TRACE_RE = re.compile(
    r"serve_trace:\s+"
    r"(?:inst=(?P<inst>\S+)\s+)?"
    r"(?:run=(?P<run>\S+)\s+)?"
    r"(?:seq=(?P<seq>\d+)\s+)?"
    r"mode=(?P<mode>\S+)\s+"
    r"bs=(?P<bs>\S+)\s+"
    r"spec_rounds=(?P<spec_rounds>\d+)\s+"
    r"spec_committed=(?P<spec_committed>\d+)\s+"
    r"plain_steps=(?P<plain_steps>\d+)\s+"
    r"drafted=(?P<drafted>\d+)\s+"
    r"accepted=(?P<accepted>\d+)\s+"
    r"spec_accept=(?P<spec_accept>[\d.]+)%\s+"
    r"spec_commit=(?P<spec_commit>[\d.]+)/round\s+"
    r"cuda_decode_graph=capture:(?P<decode_cap>\d+),replay:(?P<decode_rep>\d+)\s+"
    r"cuda_prefill_graph=capture:(?P<prefill_cap>\d+),replay:(?P<prefill_rep>\d+)\s+"
    r"prefill_ms=(?P<prefill_ms>[\d.]+)\s+"
    r"decode_ms=(?P<decode_ms>[\d.]+)\s+"
    r"stand_down=(?P<stand_down>\S+)"
    r"(?:\s+status=(?P<status>\S+))?"
)

VALID_STATUSES = {"completed", "canceled", "error"}


@dataclass
class TraceRecord:
    source: str
    mode: str
    bs: str
    spec_rounds: int
    spec_committed: int
    plain_steps: int
    drafted: int
    accepted: int
    spec_accept_pct: float
    spec_commit_per_round: float
    cuda_decode_graph_capture: int
    cuda_decode_graph_replay: int
    cuda_prefill_graph_capture: int
    cuda_prefill_graph_replay: int
    prefill_ms: float
    decode_ms: float
    stand_down: str
    status: str = "completed"
    run_id: str | None = None
    seq_id: int | None = None
    prompt_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def total_ms(self) -> float:
        return self.prefill_ms + self.decode_ms

    @property
    def estimated_completion_tokens(self) -> int:
        if self.completion_tokens is not None:
            return self.completion_tokens
        decoded = self.spec_committed + self.plain_steps
        return decoded + 1 if (decoded > 0 or self.prefill_ms > 0.0) else 0

    @property
    def decode_tok_s(self) -> float | None:
        toks = self.estimated_completion_tokens
        if self.decode_ms <= 0.0 or toks <= 1:
            return None
        return toks / (self.decode_ms / 1000.0)

    @property
    def is_speculative_eligible(self) -> bool:
        return self.spec_rounds > 0 or self.drafted > 0 or "dflash" in self.mode or "mtp" in self.mode or "spec" in self.mode or "->" in self.mode

    @property
    def stood_down(self) -> bool:
        return self.stand_down != "none"


def percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (p / 100.0) * (len(sorted_vals) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


API_IDENTITY_RE = re.compile(
    r"(?:\sinst=(?P<inst>\S+))?(?:\srun=(?P<run>\S+))?\sseq=(?P<seq>\d+)\s*$"
)


def _fmt_key(key: tuple[str | None, str | None, int]) -> str:
    inst_id, run_id, seq_id = key
    parts = []
    if inst_id:
        parts.append(f"inst={inst_id}")
    if run_id:
        parts.append(f"run={run_id}")
    parts.append(f"seq={seq_id}")
    return " ".join(parts)


def _trace_record(
    source: str,
    tr_m: re.Match[str],
    api: dict[str, int] | None,
    run_id: str | None = None,
    seq_id: int | None = None,
) -> TraceRecord:
    return TraceRecord(
        source=source,
        mode=tr_m.group("mode"),
        bs=tr_m.group("bs"),
        spec_rounds=int(tr_m.group("spec_rounds")),
        spec_committed=int(tr_m.group("spec_committed")),
        plain_steps=int(tr_m.group("plain_steps")),
        drafted=int(tr_m.group("drafted")),
        accepted=int(tr_m.group("accepted")),
        spec_accept_pct=float(tr_m.group("spec_accept")),
        spec_commit_per_round=float(tr_m.group("spec_commit")),
        cuda_decode_graph_capture=int(tr_m.group("decode_cap")),
        cuda_decode_graph_replay=int(tr_m.group("decode_rep")),
        cuda_prefill_graph_capture=int(tr_m.group("prefill_cap")),
        cuda_prefill_graph_replay=int(tr_m.group("prefill_rep")),
        prefill_ms=float(tr_m.group("prefill_ms")),
        decode_ms=float(tr_m.group("decode_ms")),
        stand_down=tr_m.group("stand_down"),
        status=tr_m.group("status") or "completed",
        run_id=run_id,
        seq_id=seq_id,
        prompt_tokens=api["prompt"] if api else None,
        cached_prompt_tokens=api["cached"] if api else None,
        completion_tokens=api["generated"] if api else None,
    )


def parse_log_lines_with_anomalies(
    lines: list[str], source: str = "<stdin>", allow_legacy_pairing: bool = False
) -> tuple[list[TraceRecord], list[str]]:
    """Parse records, pairing `serve_api:` with `serve_trace:` by `(inst, run, seq)` identity."""
    records: list[TraceRecord] = []
    anomalies: list[str] = []
    pending_by_key: dict[tuple[str | None, str | None, int], tuple[int, dict[str, int]]] = {}
    seen_completed_keys: set[tuple[str | None, str | None, int]] = set()
    legacy_pending: tuple[int, dict[str, int]] | None = None
    legacy_pairs = 0

    for line_no, line in enumerate(lines, start=1):
        api_m = SERVE_API_RE.search(line)
        if api_m:
            api = {
                "generated": int(api_m.group("generated")),
                "prompt": int(api_m.group("prompt")),
                "cached": int(api_m.group("cached")) if api_m.group("cached") else 0,
            }
            id_m = API_IDENTITY_RE.search(line.rstrip())
            if id_m:
                key = (id_m.group("inst"), id_m.group("run"), int(id_m.group("seq")))
                if key in pending_by_key:
                    prev_line, _ = pending_by_key[key]
                    anomalies.append(
                        f"{source}:{prev_line}: serve_api {_fmt_key(key)} has no serve_trace before "
                        f"another serve_api reused {_fmt_key(key)} at line {line_no}"
                    )
                elif key in seen_completed_keys:
                    anomalies.append(
                        f"{source}:{line_no}: duplicate request identity {_fmt_key(key)} within log"
                    )
                pending_by_key[key] = (line_no, api)
            else:
                if legacy_pending is not None:
                    anomalies.append(
                        f"{source}:{legacy_pending[0]}: serve_api has no serve_trace (no seq id)"
                    )
                legacy_pending = (line_no, api)
            continue

        tr_m = SERVE_TRACE_RE.search(line)
        if not tr_m:
            continue
        if tr_m.group("seq") is not None:
            key = (tr_m.group("inst"), tr_m.group("run"), int(tr_m.group("seq")))
            hit = pending_by_key.pop(key, None)
            if hit is None:
                if legacy_pending is not None:
                    anomalies.append(
                        f"{source}:{line_no}: serve_trace {_fmt_key(key)} pairs only with a serve_api "
                        f"without a seq id (line {legacy_pending[0]}); mixed-version log"
                    )
                    legacy_pending = None
                else:
                    anomalies.append(
                        f"{source}:{line_no}: serve_trace {_fmt_key(key)} has no serve_api with that identity"
                    )
            else:
                seen_completed_keys.add(key)
            records.append(
                _trace_record(source, tr_m, hit[1] if hit else None, key[1], key[2])
            )
        else:
            if legacy_pending is None:
                anomalies.append(f"{source}:{line_no}: serve_trace (no seq id) has no preceding serve_api")
                records.append(_trace_record(source, tr_m, None))
            else:
                records.append(_trace_record(source, tr_m, legacy_pending[1]))
                legacy_pending = None
                legacy_pairs += 1

    for key, (line_no, _) in sorted(pending_by_key.items(), key=lambda kv: kv[1][0]):
        anomalies.append(
            f"{source}:{line_no}: serve_api {_fmt_key(key)} has no serve_trace (completed request without a final trace)"
        )
    if legacy_pending is not None:
        anomalies.append(
            f"{source}:{legacy_pending[0]}: serve_api has no serve_trace (completed request without a final trace)"
        )
    if legacy_pairs and not allow_legacy_pairing:
        anomalies.append(
            f"{source}: {legacy_pairs} request(s) paired by log order (no seq ids); identity unverified, "
            "pass --allow-legacy-pairing to accept for pre-seq logs"
        )
    return records, anomalies


def parse_log_lines(lines: list[str], source: str = "<stdin>") -> list[TraceRecord]:
    return parse_log_lines_with_anomalies(lines, source, allow_legacy_pairing=True)[0]


def summarize_group(records: list[TraceRecord]) -> dict[str, Any]:
    ttfts = sorted(r.prefill_ms for r in records)
    totals = sorted(r.total_ms for r in records)
    tok_rates = sorted(r.decode_tok_s for r in records if r.decode_tok_s is not None)

    plain_steps = sum(r.plain_steps for r in records)
    dec_cap = sum(r.cuda_decode_graph_capture for r in records)
    dec_rep = sum(r.cuda_decode_graph_replay for r in records)
    pre_cap = sum(r.cuda_prefill_graph_capture for r in records)
    pre_rep = sum(r.cuda_prefill_graph_replay for r in records)

    spec_rounds = sum(r.spec_rounds for r in records)
    spec_committed = sum(r.spec_committed for r in records)
    drafted = sum(r.drafted for r in records)
    accepted = sum(r.accepted for r in records)
    spec_eligible = sum(1 for r in records if r.is_speculative_eligible)
    stand_downs = sum(1 for r in records if r.stood_down)
    stand_down_reasons = Counter(
        r.stand_down.split("(", 1)[0] for r in records if r.stood_down
    )

    graph_replay_rate_pct = (
        round(100.0 * dec_rep / plain_steps, 2) if plain_steps > 0 else None
    )
    stand_down_rate_pct = (
        round(100.0 * stand_downs / spec_eligible, 2) if spec_eligible > 0 else 0.0
    )

    return {
        "requests": len(records),
        "ttft_p50_ms": round_opt(percentile(ttfts, 50)),
        "ttft_p95_ms": round_opt(percentile(ttfts, 95)),
        "decode_tok_s_p50": round_opt(percentile(tok_rates, 50)),
        "decode_tok_s_p95": round_opt(percentile(tok_rates, 95)),
        "decode_tok_s_mean": round_opt(sum(tok_rates) / len(tok_rates)) if tok_rates else None,
        "total_latency_p50_ms": round_opt(percentile(totals, 50)),
        "total_latency_p95_ms": round_opt(percentile(totals, 95)),
        "plain_steps_total": plain_steps,
        "cuda_decode_graph_captures": dec_cap,
        "cuda_decode_graph_replays": dec_rep,
        "cuda_decode_graph_replay_rate_pct": graph_replay_rate_pct,
        "cuda_prefill_graph_captures": pre_cap,
        "cuda_prefill_graph_replays": pre_rep,
        "spec_rounds_total": spec_rounds,
        "spec_committed_total": spec_committed,
        "drafted_total": drafted,
        "accepted_total": accepted,
        "spec_accept_rate_pct": round(100.0 * accepted / drafted, 2) if drafted > 0 else None,
        "spec_commit_per_round": round(spec_committed / spec_rounds, 2) if spec_rounds > 0 else None,
        "spec_eligible_requests": spec_eligible,
        "stand_down_count": stand_downs,
        "stand_down_rate_pct": stand_down_rate_pct,
        "stand_down_reasons": dict(stand_down_reasons),
    }


def round_opt(v: float | None, digits: int = 2) -> float | None:
    return None if v is None else round(v, digits)


def expected_completion_tokens(r: TraceRecord) -> int:
    """Token accounting convention, each emitted token counted exactly once.

    - The first output token is sampled from the prompt (prefill) forward. It is never a
      plain decode step (`record_plain_decode_step` skips `is_prompt`) and never a
      speculative token, so it is the implicit `+1`.
    - Every later plain single-token step adds 1 to `plain_steps`, including the anchor a
      speculative bootstrap samples (`driver.rs`) and every step after a stand-down.
    - Every token a verify round emits adds 1 to `spec_committed`: each accepted draft and the
      correction / continuation token, counted before `finish_or_add_toks_to_seq` so a round
      that finishes the request (EOS, max_tokens, stop string) is still counted.
    - An EOS token counts like any other token if the usage line counts it.
    """
    decoded = r.plain_steps + r.spec_committed
    if r.completion_tokens == 0:
        return decoded
    return 1 + decoded


def check_correctness_and_integrity(
    all_records: list[TraceRecord],
    correctness_payload: list[dict[str, Any]] | dict[str, Any] | None = None,
    pairing_anomalies: list[str] | None = None,
) -> dict[str, Any]:
    """Keep correctness checks completely separate from throughput metrics.

    Every inconsistency is listed and fails qualification; none is averaged away.
    """
    trace_anomalies: list[str] = list(pairing_anomalies or [])
    for i, r in enumerate(all_records):
        if r.status not in VALID_STATUSES:
            trace_anomalies.append(
                f"{r.source}#{i} ({r.mode}): unknown terminal status={r.status!r}"
            )
        if r.completion_tokens is not None:
            expected = expected_completion_tokens(r)
            if r.status == "completed":
                if r.completion_tokens != expected:
                    trace_anomalies.append(
                        f"{r.source}#{i} ({r.mode}): completion_tokens={r.completion_tokens} != "
                        f"{'' if r.completion_tokens == 0 else '1 + '}plain_steps({r.plain_steps}) + "
                        f"spec_committed({r.spec_committed}) = {expected}"
                    )
            else:
                upper = 1 + r.plain_steps + r.spec_committed
                if r.completion_tokens > upper:
                    trace_anomalies.append(
                        f"{r.source}#{i} ({r.mode}, status={r.status}): "
                        f"completion_tokens={r.completion_tokens} exceeds upper bound {upper}"
                    )
        if r.spec_committed > 0 and r.spec_rounds == 0:
            trace_anomalies.append(
                f"{r.source}#{i} ({r.mode}): spec_committed={r.spec_committed} with spec_rounds=0"
            )
        if r.accepted > r.drafted:
            trace_anomalies.append(
                f"{r.source}#{i} ({r.mode}): accepted={r.accepted} > drafted={r.drafted}"
            )
        if r.spec_rounds > 0 and r.spec_committed > r.accepted + r.spec_rounds:
            trace_anomalies.append(
                f"{r.source}#{i} ({r.mode}): spec_committed={r.spec_committed} > accepted({r.accepted}) + one token per round({r.spec_rounds})"
            )
        if r.cuda_decode_graph_replay > r.plain_steps:
            trace_anomalies.append(
                f"{r.source}#{i} ({r.mode}): decode_replays={r.cuda_decode_graph_replay} > plain_steps={r.plain_steps}"
            )
        if r.stood_down and not r.is_speculative_eligible:
            trace_anomalies.append(
                f"{r.source}#{i} ({r.mode}): stand_down={r.stand_down} without speculative activity"
            )

    external_checks: dict[str, Any] | None = None
    if correctness_payload is not None:
        if isinstance(correctness_payload, dict) and "cases" in correctness_payload:
            cases = correctness_payload["cases"]
        elif isinstance(correctness_payload, list):
            cases = correctness_payload
        else:
            cases = [correctness_payload]
        passed = sum(1 for c in cases if c.get("passed") is True or c.get("exact") is True)
        failed = sum(1 for c in cases if c.get("passed") is False or c.get("exact") is False)
        external_checks = {
            "total": len(cases),
            "passed": passed,
            "failed": failed,
            "all_passed": failed == 0 and len(cases) > 0,
        }

    return {
        "trace_records_checked": len(all_records),
        "trace_invariant_anomalies": trace_anomalies,
        "trace_invariants_ok": len(trace_anomalies) == 0,
        "external_correctness": external_checks,
    }


def detect_regressions(
    current_modes: dict[str, dict[str, Any]],
    baseline_modes: dict[str, dict[str, Any]],
    max_ttft_regression_pct: float,
    max_toks_drop_pct: float,
    max_total_regression_pct: float,
) -> list[str]:
    regressions: list[str] = []
    for mode, cur in sorted(current_modes.items()):
        base = baseline_modes.get(mode)
        if not base:
            continue
        for field, limit, label in [
            ("ttft_p50_ms", max_ttft_regression_pct, "TTFT p50"),
            ("ttft_p95_ms", max_ttft_regression_pct, "TTFT p95"),
            ("total_latency_p50_ms", max_total_regression_pct, "total latency p50"),
            ("total_latency_p95_ms", max_total_regression_pct, "total latency p95"),
        ]:
            c_val = cur.get(field)
            b_val = base.get(field)
            if c_val and b_val and b_val > 0:
                delta_pct = 100.0 * (c_val - b_val) / b_val
                if delta_pct > limit:
                    regressions.append(
                        f"mode={mode} {label} regressed +{delta_pct:.1f}% ({b_val}ms -> {c_val}ms > +{limit:.1f}%)"
                    )
        c_tps = cur.get("decode_tok_s_p50")
        b_tps = base.get("decode_tok_s_p50")
        if c_tps is not None and b_tps and b_tps > 0:
            drop_pct = 100.0 * (b_tps - c_tps) / b_tps
            if drop_pct > max_toks_drop_pct:
                regressions.append(
                    f"mode={mode} decode_tok_s p50 dropped -{drop_pct:.1f}% ({b_tps} -> {c_tps} tok/s > -{max_toks_drop_pct:.1f}%)"
                )
    return regressions


def aggregate_records(
    records: list[TraceRecord],
    min_completion_tokens: int = 8,
    min_prompt_tokens: int = 0,
    correctness_payload: list[dict[str, Any]] | dict[str, Any] | None = None,
    pairing_anomalies: list[str] | None = None,
) -> dict[str, Any]:
    measured: list[TraceRecord] = []
    excluded_probes: list[TraceRecord] = []
    non_completed: list[TraceRecord] = []
    status_counts = Counter(r.status for r in records)

    for r in records:
        if r.status != "completed":
            non_completed.append(r)
            continue
        toks = r.estimated_completion_tokens
        ptoks = r.prompt_tokens if r.prompt_tokens is not None else min_prompt_tokens
        if toks < min_completion_tokens or ptoks < min_prompt_tokens:
            excluded_probes.append(r)
        else:
            measured.append(r)

    by_mode: dict[str, list[TraceRecord]] = defaultdict(list)
    for r in measured:
        by_mode[r.mode].append(r)

    modes_summary = {
        mode: summarize_group(group) for mode, group in sorted(by_mode.items())
    }

    return {
        "lifecycle": {
            "total_records": len(records),
            "completed": status_counts.get("completed", 0),
            "canceled": status_counts.get("canceled", 0),
            "error": status_counts.get("error", 0),
            "by_status": dict(sorted(status_counts.items())),
        },
        "throughput": {
            "min_completion_tokens_filter": min_completion_tokens,
            "min_prompt_tokens_filter": min_prompt_tokens,
            "measured_requests": len(measured),
            "excluded_probe_requests": len(excluded_probes),
            "excluded_non_completed_requests": len(non_completed),
            "overall": summarize_group(measured),
            "by_mode": modes_summary,
        },
        "correctness": check_correctness_and_integrity(
            records, correctness_payload, pairing_anomalies
        ),
    }


def format_markdown(report: dict[str, Any]) -> str:
    tp = report["throughput"]
    corr = report["correctness"]
    lines = [
        f"Measured requests: {tp['measured_requests']} (excluded short/probe requests: {tp['excluded_probe_requests']})",
        "",
        "| Mode | N | TTFT p50 (ms) | TTFT p95 (ms) | Decode p50 (tok/s) | Graph replays (dec) | Replay rate | Spec accept | Stand-down rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, m in sorted(tp["by_mode"].items()):
        fmt = lambda v, suf="": "-" if v is None else f"{v}{suf}"
        lines.append(
            f"| `{mode}` | {m['requests']} | {fmt(m['ttft_p50_ms'])} | {fmt(m['ttft_p95_ms'])} | "
            f"{fmt(m['decode_tok_s_p50'])} | {m['cuda_decode_graph_replays']} | "
            f"{fmt(m['cuda_decode_graph_replay_rate_pct'], '%')} | "
            f"{fmt(m['spec_accept_rate_pct'], '%')} | {fmt(m['stand_down_rate_pct'], '%')} |"
        )
    lines.append("")
    lines.append(
        f"Correctness/trace invariants: ok={corr['trace_invariants_ok']} "
        f"(checked={corr['trace_records_checked']}, anomalies={len(corr['trace_invariant_anomalies'])})"
    )
    if corr["trace_invariant_anomalies"]:
        for a in corr["trace_invariant_anomalies"]:
            lines.append(f"  - anomaly: {a}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="Server log files containing serve_trace lines")
    parser.add_argument("--min-completion-tokens", type=int, default=8)
    parser.add_argument("--min-prompt-tokens", type=int, default=0)
    parser.add_argument("--correctness-json", help="Optional JSON file with separate output-correctness results")
    parser.add_argument("--baseline-json", help="Optional baseline report JSON to check regressions against")
    parser.add_argument("--max-ttft-regression-pct", type=float, default=15.0)
    parser.add_argument("--max-toks-drop-pct", type=float, default=10.0)
    parser.add_argument("--max-total-regression-pct", type=float, default=15.0)
    parser.add_argument("--json-out", help="Write JSON report to this path")
    parser.add_argument(
        "--allow-legacy-pairing",
        action="store_true",
        help="Accept order-based pairing for logs written before serve lines carried seq=<id>",
    )
    args = parser.parse_args(argv)

    all_records: list[TraceRecord] = []
    pairing: list[str] = []
    for path in args.logs:
        with open(path, "r", errors="replace") as f:
            recs, anomalies = parse_log_lines_with_anomalies(
                f.readlines(),
                source=os.path.basename(path),
                allow_legacy_pairing=args.allow_legacy_pairing,
            )
        all_records.extend(recs)
        pairing.extend(anomalies)

    corr_payload = None
    if args.correctness_json:
        with open(args.correctness_json) as f:
            corr_payload = json.load(f)

    report = aggregate_records(
        all_records,
        min_completion_tokens=args.min_completion_tokens,
        min_prompt_tokens=args.min_prompt_tokens,
        correctness_payload=corr_payload,
        pairing_anomalies=pairing,
    )

    regressions: list[str] = []
    if args.baseline_json:
        with open(args.baseline_json) as f:
            baseline = json.load(f)
        base_modes = baseline.get("throughput", baseline).get("by_mode", {})
        regressions = detect_regressions(
            report["throughput"]["by_mode"],
            base_modes,
            args.max_ttft_regression_pct,
            args.max_toks_drop_pct,
            args.max_total_regression_pct,
        )
    report["regressions"] = regressions

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)

    print(format_markdown(report))
    if regressions:
        print("\nREGRESSIONS DETECTED:")
        for r in regressions:
            print(f"  - {r}")
        return 2
    if not report["correctness"]["trace_invariants_ok"]:
        return 3
    ext = report["correctness"]["external_correctness"]
    if ext is not None and not ext["all_passed"]:
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
