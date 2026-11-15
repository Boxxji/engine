// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: MIT

//! Loopback benchmark for `EpochFencedKvDirectory` and `KvBlockTransport`.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use hanzo_kv_fabric::{
    compute_block_digest, kv_block_hash, EpochFencedKvDirectory, InMemoryKvBackend, KvBlockMeta,
    KvBlockTransport,
};

fn percentile_us(samples: &mut [Duration], pct: f64) -> f64 {
    samples.sort_unstable();
    let idx = (((samples.len() - 1) as f64) * pct).round() as usize;
    samples[idx].as_secs_f64() * 1_000_000.0
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = EpochFencedKvDirectory::new();
    let backend = Arc::new(InMemoryKvBackend::new(1));
    let transport =
        KvBlockTransport::bind("127.0.0.1:0".parse::<SocketAddr>()?, Arc::clone(&backend)).await?;
    let peer = transport.local_addr();

    // 1. Directory publish + lookup + fence benchmark
    let hashes: Vec<u64> = (0u32..512)
        .map(|i| kv_block_hash(&[i, i + 1, i + 2, i + 3], 0))
        .collect();
    let empty_digest = compute_block_digest(b"bench-meta");
    let metas: Vec<KvBlockMeta> = hashes
        .iter()
        .map(|&h| KvBlockMeta {
            block_hash: h,
            num_bytes: 262_144,
            sha256: empty_digest,
        })
        .collect();

    let mut dir_times = Vec::with_capacity(50);
    for epoch in 1u64..=50 {
        let t0 = Instant::now();
        dir.publish_blocks("worker-0", epoch, peer, &metas)?;
        let owners = dir.lookup_fresh_owners(hashes[256]);
        assert_eq!(owners.len(), 1);
        let _ = dir.fence_epoch("worker-0", epoch + 1);
        dir_times.push(t0.elapsed());
    }
    println!(
        "directory publish+lookup+fence (512 blocks): p50={:.1}us p95={:.1}us",
        percentile_us(&mut dir_times, 0.50),
        percentile_us(&mut dir_times, 0.95)
    );

    // 2. Framed TCP push + pull benchmark (256 KiB and 1 MiB KV pages)
    backend.set_epoch(100);
    for &block_bytes in &[256usize * 1024, 1024 * 1024] {
        let payload: Vec<u8> = (0..block_bytes).map(|i| (i & 0xFF) as u8).collect();
        let hash = kv_block_hash(&[42, block_bytes as u32], 1);

        // Warmup
        KvBlockTransport::<InMemoryKvBackend>::push_block(peer, hash, 100, &payload).await?;
        let (_, got) = KvBlockTransport::<InMemoryKvBackend>::pull_block(peer, hash, 100).await?;
        assert_eq!(got.len(), block_bytes);

        let iters = 30;
        let mut push_times = Vec::with_capacity(iters);
        let mut pull_times = Vec::with_capacity(iters);
        for i in 0..iters {
            let epoch = 100 + i as u64;
            let t0 = Instant::now();
            KvBlockTransport::<InMemoryKvBackend>::push_block(peer, hash, epoch, &payload).await?;
            push_times.push(t0.elapsed());

            let t1 = Instant::now();
            let (_ep, bytes) =
                KvBlockTransport::<InMemoryKvBackend>::pull_block(peer, hash, epoch).await?;
            pull_times.push(t1.elapsed());
            assert_eq!(bytes.len(), block_bytes);
        }
        let push_p50 = percentile_us(&mut push_times, 0.50);
        let pull_p50 = percentile_us(&mut pull_times, 0.50);
        let mib = block_bytes as f64 / (1024.0 * 1024.0);
        println!(
            "HKV1 loopback {} KiB: push p50={push_p50:.1}us ({:.1} MiB/s) | pull p50={pull_p50:.1}us ({:.1} MiB/s)",
            block_bytes / 1024,
            mib / (push_p50 / 1_000_000.0),
            mib / (pull_p50 / 1_000_000.0),
        );
    }

    Ok(())
}
