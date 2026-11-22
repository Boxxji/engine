#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_engine::{
    active_kv_phase_policy, evaluate_multiturn_hybrid_prefix_reuse,
    evaluate_phase_aware_kv_precision, set_kv_phase_policy_override,
    verify_hybrid_prefix_provenance_isolation, HybridEvalConfig, KvPhasePolicy,
};
use hanzo_ml::{DType, Device, Result};
use serde::Serialize;

const ENV_ENGINE_EVAL_OUT_JSON: &str = "ENGINE_EVAL_OUT_JSON";

#[derive(Serialize)]
struct EngineTrack3EvalBundle {
    provenance_isolation: hanzo_engine::ProvenanceIsolationReport,
    hybrid_multiturn_f32: hanzo_engine::HybridMultiturnEvalReport,
    hybrid_multiturn_bf16: hanzo_engine::HybridMultiturnEvalReport,
    kv_phase_eval: hanzo_engine::KvPhaseEvalReport,
}

#[test]
fn hybrid_prefix_and_kv_phase_cpu_gate() -> Result<()> {
    set_kv_phase_policy_override(Some(KvPhasePolicy::PhaseAware {
        min_fp8_decode_ctx: 2048,
    }));
    assert_eq!(
        active_kv_phase_policy(),
        KvPhasePolicy::PhaseAware {
            min_fp8_decode_ctx: 2048
        }
    );

    let dev = Device::Cpu;
    let iso = verify_hybrid_prefix_provenance_isolation(&dev)?;
    assert_eq!(iso.owner_hit_tokens, 80);
    assert_eq!(iso.cross_tenant_hits, 0);
    assert_eq!(iso.unsalted_hits_against_salted, 0);
    assert_eq!(iso.cross_model_hits, 0);
    assert_eq!(iso.cross_adapter_hits, 0);
    assert_eq!(iso.base_hits_against_adapter, 0);
    assert_eq!(iso.boundary_collision_hits, 0);
    assert_eq!(iso.disabled_env_hits, 0);
    assert_eq!(iso.post_mutation_checkpoint_rel_err, 0.0);

    let cfg = HybridEvalConfig {
        num_gdn_layers: 3,
        num_attn_layers: 1,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 64,
        hidden_size: 256,
        system_tokens: 16,
        repo_tokens: 32,
        tool_schema_tokens: 16,
        user_turn_tokens: 16,
        assistant_turn_tokens: 8,
        num_turns: 5,
    };
    let rep = evaluate_multiturn_hybrid_prefix_reuse(&dev, DType::F32, &cfg)?;
    for t in rep
        .shared_base_turns
        .iter()
        .chain(&rep.cumulative_coding_turns)
    {
        assert!(t.attn_out_max_rel < 1e-5);
        assert!(t.gdn_out_max_rel < 1e-5);
        assert!(t.gdn_state_max_rel < 1e-5);
        assert_eq!(t.conv_state_max_rel, 0.0);
    }

    let kv_rep = evaluate_phase_aware_kv_precision(&dev, &[64, 128], &[256, 512])?;
    assert_eq!(kv_rep.points.len(), 4);
    for pt in &kv_rep.points {
        assert!(pt.fp8_decode_cosine_vs_f32 > 0.999);
    }
    set_kv_phase_policy_override(None);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_hybrid_prefix_and_phase_aware_kv_eval() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    set_kv_phase_policy_override(Some(KvPhasePolicy::PhaseAware {
        min_fp8_decode_ctx: 2048,
    }));

    let iso = verify_hybrid_prefix_provenance_isolation(&dev)?;
    assert_eq!(iso.owner_hit_tokens, 80);
    assert_eq!(iso.cross_tenant_hits, 0);
    assert_eq!(iso.unsalted_hits_against_salted, 0);
    assert_eq!(iso.cross_model_hits, 0);
    assert_eq!(iso.cross_adapter_hits, 0);
    assert_eq!(iso.base_hits_against_adapter, 0);
    assert_eq!(iso.boundary_collision_hits, 0);
    assert_eq!(iso.disabled_env_hits, 0);
    assert_eq!(iso.post_mutation_checkpoint_rel_err, 0.0);

    let cfg = HybridEvalConfig::default();
    let rep_f32 = evaluate_multiturn_hybrid_prefix_reuse(&dev, DType::F32, &cfg)?;
    for t in rep_f32
        .shared_base_turns
        .iter()
        .chain(&rep_f32.cumulative_coding_turns)
    {
        assert!(
            t.attn_out_max_rel < 1e-5,
            "CUDA F32 turn {} ({}) attn rel_err {} >= 1e-5",
            t.turn,
            t.mode,
            t.attn_out_max_rel
        );
        assert!(
            t.gdn_out_max_rel < 1e-5,
            "CUDA F32 turn {} ({}) gdn rel_err {} >= 1e-5",
            t.turn,
            t.mode,
            t.gdn_out_max_rel
        );
        assert!(
            t.gdn_state_max_rel < 1e-5,
            "CUDA F32 turn {} ({}) gdn_state rel_err {} >= 1e-5",
            t.turn,
            t.mode,
            t.gdn_state_max_rel
        );
        assert_eq!(t.conv_state_max_rel, 0.0);
    }

    let rep_bf16 = evaluate_multiturn_hybrid_prefix_reuse(&dev, DType::BF16, &cfg)?;
    for t in rep_bf16
        .shared_base_turns
        .iter()
        .chain(&rep_bf16.cumulative_coding_turns)
    {
        assert!(t.attn_out_cosine > 0.9999);
        assert!(t.gdn_state_max_rel < 1e-5);
        assert_eq!(t.conv_state_max_rel, 0.0);
    }

    let kv_rep =
        evaluate_phase_aware_kv_precision(&dev, &[64, 128, 256], &[4096, 8192, 16384, 32768])?;
    for pt in &kv_rep.points {
        assert!(
            pt.fp8_decode_cosine_vs_f32 > 0.999,
            "d={} ctx={} fp8 cosine {}",
            pt.head_dim,
            pt.context_len,
            pt.fp8_decode_cosine_vs_f32
        );
        assert!(
            pt.phase_aware_prefill_max_rel <= pt.fp8_always_prefill_max_rel,
            "phase-aware prefill must be at least as accurate as fp8-always"
        );
    }

    eprintln!(
        "\n[CUDA Provenance Isolation] owner_hit={} cross_tenant={} cross_model={} cross_adapter={} boundary={} mut_rel={:.2e}",
        iso.owner_hit_tokens,
        iso.cross_tenant_hits,
        iso.cross_model_hits,
        iso.cross_adapter_hits,
        iso.boundary_collision_hits,
        iso.post_mutation_checkpoint_rel_err
    );
    eprintln!(
        "\n[CUDA Multi-Turn Hybrid Prefix Reuse BF16] 5-turn cumulative cold={:.2}ms warm={:.2}ms speedup={:.2}x saved_flops={:.3e} saved_kv_bytes={}",
        rep_bf16.cumulative_cold_ms,
        rep_bf16.cumulative_warm_ms,
        rep_bf16.cumulative_ttft_speedup,
        rep_bf16.total_saved_prefill_flops,
        rep_bf16.total_saved_kv_write_bytes
    );
    for t in &rep_bf16.cumulative_coding_turns {
        eprintln!(
            "  turn={} prompt={} reused={} suffix={} cold={:.2}ms warm={:.2}ms speedup={:.2}x f32_attn_rel={:.2e} bf16_attn_rel={:.2e} gdn_st_rel={:.2e} conv_rel={:.2e}",
            t.turn,
            t.total_prompt_tokens,
            t.reused_prefix_tokens,
            t.prefilled_suffix_tokens,
            t.cold_reprefill_ms,
            t.warm_prefix_reuse_ms,
            t.ttft_speedup,
            rep_f32.cumulative_coding_turns[t.turn - 1].attn_out_max_rel,
            t.attn_out_max_rel,
            t.gdn_state_max_rel,
            t.conv_state_max_rel
        );
    }

    eprintln!("\n[CUDA Phase-Aware KV Precision (BF16 vs FP8 E4M3)]");
    for pt in &kv_rep.points {
        eprintln!(
            "  d={:3} ctx={:5} | prefill_us(bf16={:.1}, pa={:.1}, fp8={:.1}) pre_rel(pa={:.2e}, fp8={:.2e}) | decode_us(bf16={:.1}, fp8={:.1}, {:.2}x) bw_gbs(bf16={:.1}, fp8={:.1}) kv36_mib(bf16={:.1}, fp8={:.1}) dec_rel(bf16={:.2e}, fp8={:.2e}) cos={:.6}",
            pt.head_dim,
            pt.context_len,
            pt.bf16_prefill_us,
            pt.phase_aware_prefill_us,
            pt.fp8_always_prefill_us,
            pt.phase_aware_prefill_max_rel,
            pt.fp8_always_prefill_max_rel,
            pt.bf16_decode_us,
            pt.fp8_decode_us,
            pt.decode_speedup,
            pt.bf16_decode_gb_s,
            pt.fp8_decode_gb_s,
            pt.bf16_kv_36l_mib,
            pt.fp8_kv_36l_mib,
            pt.bf16_decode_max_rel_vs_f32,
            pt.fp8_decode_max_rel_vs_f32,
            pt.fp8_decode_cosine_vs_f32
        );
    }

    if let Ok(out_path) = std::env::var(ENV_ENGINE_EVAL_OUT_JSON) {
        let bundle = EngineTrack3EvalBundle {
            provenance_isolation: iso,
            hybrid_multiturn_f32: rep_f32,
            hybrid_multiturn_bf16: rep_bf16,
            kv_phase_eval: kv_rep,
        };
        let json = serde_json::to_string_pretty(&bundle)
            .map_err(|e| hanzo_ml::Error::Msg(format!("json encode failed: {e}")))?;
        std::fs::write(&out_path, json)
            .map_err(|e| hanzo_ml::Error::Msg(format!("write {out_path} failed: {e}")))?;
    }

    set_kv_phase_policy_override(None);
    Ok(())
}
