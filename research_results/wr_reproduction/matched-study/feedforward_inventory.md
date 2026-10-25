# Matched PR393 feedforward inventory

UTC: 2026-10-25. Shared backend: `4cde32668b88b717ceba5db8b84017a7e44e5148f3ef2c2ba88c99c4960ddc24`.

The immutable forward contains five dense sites and five routed sites, each with eight top-1 experts. Layer 8 has both a routed branch and a parallel dense branch. Layers 7 and 9 have no feedforward block.

| Layer | Original feedforward sites | All KS3 | First MLP | Last MLP | Both MLP |
|---|---|---|---|---|---|
| 0 | dense | KS3 | MLP | KS3 | MLP |
| 1 | routed_experts | KS3 | KS3 | KS3 | KS3 |
| 2 | dense | KS3 | KS3 | KS3 | KS3 |
| 3 | routed_experts | KS3 | KS3 | KS3 | KS3 |
| 4 | dense | KS3 | KS3 | KS3 | KS3 |
| 5 | routed_experts | KS3 | KS3 | KS3 | KS3 |
| 6 | routed_experts | KS3 | KS3 | KS3 | KS3 |
| 7 | none | none | none | none | none |
| 8 | parallel_dense, routed_experts | KS3, KS3 | KS3, KS3 | KS3, KS3 | KS3, KS3 |
| 9 | none | none | none | none | none |
| 10 | dense | KS3 | KS3 | MLP | MLP |

Every dense original MLP has two 2816x768 projection matrices. Each original expert has two 1024x768 matrices. The initial KS3 recipe uses 1024 charts, rank one, two 1024x768 projections, eight geometry values per chart, GS32 quantization, FP16 nonlinearities and BF16 training products. Retained hybrid MLPs retain the original FP8 producers, formats, layouts, delayed scales and autograd operator.

The `mlp_bank` ANVIL bank now contains dense KS3 projections; its padding and replaced hybrid slots have zero per-matrix learning rate. Hybrid MLPs use a separate four-slot padded bank with the original shape-based learning rate, 2x output-projection multiplier, FP32 velocity/master mantissa, and frozen padding. Eight logical-rank contributions still accumulate before each original global update. Geometry uses replicated Adam at multiplier 0.125.

| Architecture | Total allocated model parameters, including padding | Active conventional feedforward MLP instances |
|---|---:|---:|
| ks3_all | 326980456 | 0 |
| ks3_both_mlp | 344281960 | 2 |
| ks3_first_mlp | 344281960 | 1 |
| ks3_last_mlp | 344281960 | 1 |

Attention projections, vocabulary head, embeddings, MoE routers, token normalization, CPLM, and MUDD coefficient/gate generators are unchanged. MUDD generators produce residual-mixing coefficients or gate values rather than the ordinary feedforward residual outputs being replaced. CPU audits hash every nonfeedforward parameter and verify identical seeded initialization, including routers and retained MLP matrices.

CPU audits passed for all four architectures without GPU initialization. The independent output/input/weight/geometry/post-gradient GPU oracle and real 24-update full-graph warmup remain mandatory before convergence trials. Runtime coverage is recorded separately; a CPU inventory alone is not execution or convergence evidence.
