# Reproducibility and provenance

All numbers in the paper are from registered runs in the project corpus.

## Environment
- Hardware: RTX 4060 Ti (16 GB), Ryzen 5 5600X (AVX-2); WSL2 Ubuntu.
- Software: PyTorch 2.5.1+cu121, transformers 5.6.2, Triton 3.1.0.
- Model / data: meta-llama/Llama-3.2-1B (bf16), WikiText-2 validation.
- Fixed unless stated: T_dispatch=64, k_min/k_window=64, fp32 softmax, greedy decode.

## Result → source map (canonical numbers)
- Theorem 1, 45/45 brute-force (Jaccard ≥ 0.998) ............ Paper 2 §4.5
- Theorem 3, α≈0.31, K_eff/N≈0.4% @20K ..................... Paper 2 §3, Fig 1
- Quality scaling Table (hero +0.308% @20K c=0.5) .......... Paper 2 Table 1
- top-k vs rank-window 3.06–3.32× ......................... Paper 2 Table 2
- Pareto sweet spot +0.118% @8K c=0.5 ..................... Paper 2 Table 3
- PCA collapse 149×→438×, latency 5×→36× .................. Paper 2 §4.3
- Hero ΔNLL z=5.32, p≪1e-6, n=19,936 ...................... Paper 2 §4.4
- Memory wall (B=8 → 16 GB) ............................... Paper 2 §5.5 / Track 3
- Per-pass profile, Pass-2 = 74.4% M4 (L4) ................ phase_8_1a Step 0
- ABKV mode A 1.21× SDPA = 95.4% of 1.269× ceiling ........ phase_8_1a §1.5
- Matched-control NULL: NLL diff +2.72e-4, p=0.26 ......... matched_controls Table 1
- Per-query certificate, 0 FN / 45 cells ................. Φ-Transport v2.2 Lemma 6.1
- ε² = ΣγγK, V-Gram cancellation, 0.5% agreement .......... Φ-Transport v2.2 §4
- Multi-axis r*=⌈1.142 d_eff−1.048⌉, Pearson 0.9925 ....... Φ-Transport v2.2 §5.7
- Retention gate v3 ROBUST, 0 FAIL / 10 cells @c=0.15 ..... RETENTION_STUDY §3
- Leakage out_corr/correct up to 0.70 .................... RETENTION_STUDY §4 (F2)
- Sort artifact 21.6× / 2,800 ops / 50/50 determinism ..... Φ v2.2 §8 / DCR_stress

## Deferred (must precede the boundary claim)
M1 multi-seed boundary; M2 inject (starvation vs mis-selection); M3 anti-inject;
M4 TTFT; Gap-1 matched-control of Theorem 3; cross-model α.
