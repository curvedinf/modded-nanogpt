# Corrected fixed-text probe protocol

Screening uses 1048576 fixed tokens in8192-token chunks every 120 elapsed seconds. Original full-vocabulary BF16 CPLM, raw weights and fixed640/1408 YaRN state remain identical across architectures and stages. The900-second clock includes all setup, scoring, restoration, screening and diagnostics. Final full10485760-token scoring is separate; its setup remains timed.

The bounded paired calibration uses the same seed169 original training prefix near120 seconds on both architectures. See calibration-gates.json for actual costs,8192-token block confidence intervals and contiguous32768-token sensitivity. Cost projections include20% headroom but do not replace measured total-run overhead. This is not a universal precision guarantee; inspect actual compatible paired observations before placement release.

V1 curves remain historical evidence and are excluded from this registry. Exactly one corrected MLP169 and KS3-experts169 reference are acquired. A qualifying KS3 triggers three fresh unpruned seeds369/370/371. Full confirmation uses a fresh262144-token interval starting at16777216, disjoint from V1 holdout, screening and full validation; it is not a tuning signal. Below5, first one-sided inclusive+.01, latest nonfuture within3seconds and exemption rules are unchanged.

## Verified winning configuration

The current verified winner replaces all 40 routed MoE expert MLPs with **1024-wide rank-one KS3**, using GS32/BF16 products, FP16 nonlinearities, zero initial geometry coefficients, geometry Adam multiplier0.125 with betas(0.9,0.99), and ReLU-squared output features. All five original dense MLP branches remain, at width2816. The shared attention/MLP/head FP8 backend and original PR393 model-wide settings match the control.

Fresh seeds369/370/371 reached full-validation losses **3.3023710251, 3.3023450375, and 3.3026838303** at897.7495/898.2874/898.3942 training seconds. Mean **3.3024666309**, sample SD **0.0001885484**, versus the matched MLP control **3.3339109421 at898.1671 seconds**. The mean difference is **-0.0314443111**. All three also improved the disjoint reserved confirmation interval; those scores were not used for selection.

The frozen training code is `ed40bae28e01c78801dbffacb057aa7bc76f1e58`. [Winning recipe, exact source/config references and results](../WINNING_KS3.json) · [Full three-seed verification](verification-v2-summary.json). The lower historical endpoint3.2995657921 used the earlier64K probe protocol and is recorded separately. These results establish the current expert-only hybrid; ongoing single-dense-site ablations have not yet displaced it.
