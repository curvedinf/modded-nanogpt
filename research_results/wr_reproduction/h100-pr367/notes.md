# PR367 retrieval + KS3 MoE study

## 2026-10-28 21:01 UTC — pinned reproduction and first port

The user requested an unchanged PR367 validation followed by moving KS3 research to its retrieval architecture with MoE. PR367 head is `e390cb25cf5c732ed4471c2ad861af28114e0a56`; its reported 16-run result is 6.702 +/- 0.018 s, CE 3.27406 +/- 0.00400. The PR remains open. These are author-reported values, not our reproduction.

The unchanged control started as `20261028T210058Z`. All 370 training shards and the validation shard were verified against published SHA256 hashes at the pinned dataset revision. The original 37B-token retrieval corpus, 48 build threads, 512 model width, 1792 dense width, 530 scheduled + 10 extension updates, on-clock index construction, and 10,485,760-token final validation are retained.

Hypothesis: retain PR367's retrieval, attention, head, training objective, and shared schedule, and replace layers 1, 3, 5, 6, 8 with eight-expert top-one KS3 MoE. Each expert has 1024 independent rank-one cubic charts, FP8 products, GS32 packing, FP16 nonlinearities and FP32 STE. Keep layers 0, 2, 4, 9, 10 and the parallel branch dense. Geometry Adam multiplier is 0.125 with matching tail averaging. Use PR393's stable four-map expert ANVIL at margin 1.1; shared PR367 banks keep their original six maps.

The initial port uses PR367's BF16 bank all-gather with locally refreshed FP8 weight caches and reusable dual layouts. It does not yet implement PR393's owner-only FP8 weight transport. This is an explicit implementation limitation to optimize after correctness and convergence.

512-wide block screen: all producer layouts bitwise consistent, all 41,943,040 packed weight elements matched the independent reference. Projection RMS error 0.000334 and weight-gradient RMS error 0.001798 against dequantized matrix-product oracles. Complete block median: FP8 0.471400 ms, BF16 0.533904 ms at 16,384 tokens. This is an isolated measurement, not a model speed or convergence claim.

The gradient-collector check found a noncontiguous test fixture, then a Dynamo restriction on directly calling another autograd Function's backward. The fixture was corrected and the shared backward extracted as a plain function without changing its math. The compiled check is being rerun. Failure logs are preserved under the setup log directory.

Decision: finish the exact control before any KS3 training. Require the compiled gradient-collector check to pass, then run the first full KS3-MoE candidate at seed 413. Prior PR393 study STOP markers remain in place.

## 2026-10-28 21:01 UTC — compiled gradient collector passed

The corrected collector passed eager and full-graph compiled backward checks: finite input, router, post-scale, geometry, up-projection and down-projection gradients; both projection gradients are bitwise equal to the kernel-written bank. Evidence is `gradient-collector-check.json`. The port and 512-wide checks were committed and pushed as `a82033e`. The first KS3 full run is queued behind the unchanged control, with no concurrent training.
