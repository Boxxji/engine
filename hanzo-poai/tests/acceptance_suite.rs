// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Integration, security, and concurrency acceptance tests for `hanzo-poai`.

use std::sync::{Arc, Barrier};
use std::thread;

use hanzo_poai::{
    commit_weight_matrix, os_random_seed, prove_matrix_product,
    prove_matrix_product_with_cached_b_commit, prover_matmul, verify_matrix_product,
    verify_matrix_product_with_cached_b_commit, verify_matrix_product_with_mode, ActivationMatrix,
    ChallengeMode, MatrixProofReceipt, OperatorKind, OutputMatrix, PackedQuantFormat,
    PackedWeightMatrix, PoaiError, VerificationProfile, MAX_SAFE_SIGNED, MODULUS_P,
};

fn det_u64(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn make_case(
    seed: &mut u64,
    format: PackedQuantFormat,
    m: usize,
    k: usize,
    n: usize,
) -> (ActivationMatrix, PackedWeightMatrix, OutputMatrix) {
    let a_data: Vec<i8> = (0..m * k)
        .map(|_| ((det_u64(seed) % 241) as i32 - 120) as i8)
        .collect();
    let a = ActivationMatrix::new(m, k, a_data).unwrap();
    let blocks = k / 32;
    let scales: Vec<i16> = (0..n * blocks)
        .map(|_| {
            let s = ((det_u64(seed) % 65) as i32 - 32) as i16;
            if s == 0 {
                1
            } else {
                s
            }
        })
        .collect();
    let q: Vec<i8> = match format {
        PackedQuantFormat::Int8Q8_0 => (0..k * n)
            .map(|_| ((det_u64(seed) % 241) as i32 - 120) as i8)
            .collect(),
        PackedQuantFormat::PackedInt4Q4_0 => (0..k * n)
            .map(|_| ((det_u64(seed) % 16) as i32 - 8) as i8)
            .collect(),
        PackedQuantFormat::PackedTernaryI2S => {
            (0..k * n).map(|_| (det_u64(seed) % 3) as i8 - 1).collect()
        }
    };
    let b = PackedWeightMatrix::pack_from_column_major(format, k, n, &q, &scales).unwrap();
    let c = prover_matmul(&a, &b).unwrap();
    (a, b, c)
}

#[test]
fn rejects_all_ineligible_float_and_non_matmul_operators() {
    let profile = VerificationProfile::v1();
    for op in [
        OperatorKind::Fp16Gemm,
        OperatorKind::Bf16Gemm,
        OperatorKind::Fp32Gemm,
        OperatorKind::Fp8Gemm,
        OperatorKind::FlashAttention,
        OperatorKind::Normalization,
        OperatorKind::Sampling,
        OperatorKind::StateUpdate,
    ] {
        let err = profile.ensure_eligible(op).unwrap_err();
        assert!(matches!(err, PoaiError::IneligibleOperator { .. }));
    }
}

#[test]
fn verifies_150_correct_and_detects_150_corrupted_across_formats() {
    let profile = VerificationProfile::v1();
    let mut seed = 0xCAFE_BABE_1350_u64;
    let formats = [
        PackedQuantFormat::Int8Q8_0,
        PackedQuantFormat::PackedInt4Q4_0,
        PackedQuantFormat::PackedTernaryI2S,
    ];

    for idx in 0..150 {
        let fmt = formats[idx % 3];
        let m = [1, 2, 4, 8][idx % 4];
        let k = [32, 64, 128][idx % 3];
        let n = [16, 32, 64][idx % 3];

        let (a, b, c) = make_case(&mut seed, fmt, m, k, n);
        let mode = if idx % 2 == 0 {
            ChallengeMode::FiatShamir
        } else {
            ChallengeMode::InteractiveSeed(os_random_seed().unwrap())
        };

        let receipt = prove_matrix_product(&profile, &a, &b, &c, 2, mode).unwrap();
        let decoded = MatrixProofReceipt::from_bytes(&receipt.to_bytes()).unwrap();
        verify_matrix_product_with_mode(&profile, &a, &b, &c, &decoded, mode).unwrap();

        // Corrupt C by +/-1 and re-prove with matching commit_C -> MUST fail Freivalds
        let mut c_bad = c.clone();
        let target = (det_u64(&mut seed) as usize) % c_bad.data().len();
        c_bad.data_mut()[target] += if idx % 2 == 0 { 1 } else { -1 };
        let bad_receipt = prove_matrix_product(&profile, &a, &b, &c_bad, 2, mode).unwrap();
        let res = verify_matrix_product_with_mode(&profile, &a, &b, &c_bad, &bad_receipt, mode);
        assert!(matches!(res, Err(PoaiError::FreivaldsCheckFailed { .. })));
    }
}

#[test]
fn enforces_strict_signed_field_range_no_wraparound_forgery() {
    let profile = VerificationProfile::v1();
    let mut seed = 0x1234_5678_u64;
    let (a, b, c) = make_case(&mut seed, PackedQuantFormat::Int8Q8_0, 2, 32, 16);

    // Adding MODULUS_P to an entry preserves C (mod p) algebraically, so a verifier without
    // strict range bounds would falsely accept. Verify that hanzo-poai strictly rejects it!
    let mut c_wrapped = c.clone();
    c_wrapped.data_mut()[0] = c_wrapped.data()[0].saturating_add(MODULUS_P as i64);
    assert!(c_wrapped.data()[0].unsigned_abs() > MAX_SAFE_SIGNED as u64);

    let err = prove_matrix_product(&profile, &a, &b, &c_wrapped, 2, ChallengeMode::FiatShamir)
        .unwrap_err();
    assert!(matches!(
        err,
        PoaiError::OutputOutOfFieldRange { row: 0, col: 0, .. }
    ));
}

#[test]
fn multithreaded_16_thread_concurrent_verification_is_deterministic() {
    let profile = Arc::new(VerificationProfile::v1());
    let mut seed = 0xDEAD_BEEF_1350_u64;
    let (a, b, c) = make_case(&mut seed, PackedQuantFormat::PackedInt4Q4_0, 16, 128, 64);
    let commit_b = commit_weight_matrix(&b);
    let receipt = prove_matrix_product_with_cached_b_commit(
        &profile,
        &a,
        &b,
        &c,
        2,
        ChallengeMode::FiatShamir,
        commit_b,
    )
    .unwrap();

    let a = Arc::new(a);
    let b = Arc::new(b);
    let c = Arc::new(c);
    let receipt = Arc::new(receipt);
    let barrier = Arc::new(Barrier::new(16));

    let mut handles = Vec::with_capacity(16);
    for tid in 0..16 {
        let profile = Arc::clone(&profile);
        let a = Arc::clone(&a);
        let b = Arc::clone(&b);
        let c = Arc::clone(&c);
        let receipt = Arc::clone(&receipt);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            if tid % 2 == 0 {
                verify_matrix_product(&profile, &a, &b, &c, &receipt).unwrap();
            } else {
                verify_matrix_product_with_cached_b_commit(
                    &profile, &a, &b, &c, &receipt, commit_b, None,
                )
                .unwrap();
            }
        }));
    }

    for h in handles {
        h.join().expect("verifier thread panicked");
    }
}
