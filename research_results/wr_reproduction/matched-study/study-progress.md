# Matched KS3 study progress

Snapshot: 2026-10-25T22:06:23.995193+00:00

Matched MLP full-validation losses: seed 169: 3.31410837 at 898.32s, seed 170: 3.31521773 at 897.51s, seed 171: 3.31637120 at 897.85s.
Mean 3.31523244; sample SD 0.00113149. Nominal budget: 900 timed training seconds.

KS3 discovery full-validation results: 0. Qualifying discoveries: 0.

Latest run: matched-ks3-all-20261025T220528-2ca4d7 (timing, running). Timing diagnostics are excluded from convergence and pruning references.

| Recipe | Status | Proxy-prune elapsed (s) | Proxy gap to compatible reference | Full validation |
|---|---|---:|---:|---:|
| matched-ks3-all | pruned | 384.74 | +0.03774 | Not measured |
| matched-ks3-first-mlp | error | — | — | Not measured |
| matched-ks3-first-mlp | pruned | 330.41 | +0.03398 | Not measured |
| matched-ks3-last-mlp | pruned | 330.75 | +0.03946 | Not measured |
| matched-ks3-both-mlp | pruned | 337.77 | +0.11360 | Not measured |
| matched-ks3-all-geometry025 | pruned | 327.94 | +0.03304 | Not measured |
| matched-ks3-all-projection125 | pruned | 323.47 | +0.05675 | Not measured |
| matched-ks3-all-dense1536 | pruned | 312.69 | +0.01004 | Not measured |
| matched-ks3-all-rank2 | pruned | 322.13 | +0.03740 | Not measured |
| matched-ks3-all-projection0875 | pruned | 266.87 | +0.01034 | Not measured |
| matched-ks3-all-scale050 | pruned | 323.56 | +0.03152 | Not measured |
| matched-ks3-all-rank4 | pruned | 313.23 | +0.08499 | Not measured |
| matched-ks3-all-geometry00625 | pruned | 323.79 | +0.07173 | Not measured |
| matched-ks3-all-warpminus0025 | pruned | 324.19 | +0.08099 | Not measured |
| matched-ks3-all-dense1280 | pruned | 332.39 | +0.02637 | Not measured |
| matched-ks3-all-denselr150 | pruned | 324.21 | +0.08190 | Not measured |
| matched-ks3-all-denselr075 | pruned | 324.18 | +0.07539 | Not measured |
| matched-ks3-all-dense1536-normlr | pruned | 339.63 | +0.01340 | Not measured |
| matched-ks3-all-dense2816 | pruned | 381.53 | +0.04489 | Not measured |
| matched-ks3-all-geometry100 | pruned | 248.17 | +0.03263 | Not measured |
| matched-ks3-all-geometry000 | pruned | 323.36 | +0.06496 | Not measured |
| matched-ks3-all-dense2816-fp8 | pruned | 368.48 | +0.05049 | Not measured |
| matched-ks3-all-warpplus0025 | pruned | 324.71 | +0.01009 | Not measured |
| matched-ks3-all-cubicplus0025 | pruned | 323.90 | +0.01315 | Not measured |
| matched-ks3-all-dense2816-fp8-normlr | pruned | 368.90 | +0.05097 | Not measured |
| matched-ks3-all-coupledwarp-prior | pruned | 265.57 | +0.01192 | Not measured |
| matched-ks3-all-width768-normlr | pruned | 311.20 | +0.03317 | Not measured |
| matched-ks3-all-densegeom00625 | pruned | 285.73 | +0.01051 | Not measured |
| matched-ks3-all-width1280-normlr | pruned | 345.23 | +0.08898 | Not measured |

Proxy gaps use each run’s compatible, nonfuture reference within 15 seconds. They are not validation-progress measurements and do not rank full-validation architecture quality. The earlier first-layer adapter failure remains preserved; its corrected retry is listed separately.

Frozen shared backend: `4cde32668b88b717ceba5db8b84017a7e44e5148f3ef2c2ba88c99c4960ddc24`. All three controls use full 10,485,760-token validation, raw learned weights and accepted 640/1408 evaluation state.
