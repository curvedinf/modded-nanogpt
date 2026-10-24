# KS3 research on the full MI300 model

The objective is **minimum complete time to full-validation CE ≤ 3.793633**.
Parameter count, batch sizes, optimizer steps, token budget, and learning rates are research variables.
The control is fixed; this study does not restart or lengthen the MLP run.
Every full trial continues until this target is measured, unless significant-regression pruning,
a real failure, or an explicit user stop applies. Neither a planned schedule horizon nor the
control's runtime terminates a trial. A slower target hit is measured honestly and is not a win.
The current pruning rule is `AGENTS.md` / `configs/ks3/screening_policy.json`: after the first
loss strictly below 5.0, cancel on the first compatible observation at least 0.01 worse than
the dynamic best MLP/KS3 loss at comparable elapsed time. Better candidates continue.
Run meaningful optimization changes; do not repeat an unchanged recipe merely to fill time.

## Measured control

| Measurement | MLP control |
|---|---:|
| Full FineWeb validation loss | **3.7936333656311034** |
| End-to-end wall time | **2091.555386705 s (34m 51.6s)** |
| Trainer-reported training time | 1987.480152842 s |
| Trainer-reported training plus final validation | 2077.861757079 s |
| Validation tokens | 10,485,760 |
| Training tokens | 41,082,880 |
| Peak PyTorch allocation | 142.227454 GiB |
| Hardware / seed | One MI300X VF, gfx942 / 0 |

The completed run is `logs/control-20261024-044357/`, with source archive, hashes,
dataset hashes, environment, stdout, metrics, and independent wall timing.
`configs/ks3/mlp_control.json` and `mlp_control_manifest.json` retain its measured target and provenance.
The current machine is an MI300X, not an MI100.

## Launch and resume research

```bash
python3 ks3_research/ops/researchctl.py status --brief
```

Adopt a live run. The initial trial has already run; do not duplicate it.
Launch every new trial, pilot, or smoke check through `researchctl.py start --config PATH
--label LABEL --hypothesis 'Specific change and evidence' [--kind trial|pilot|smoke]`.
The numerical tests, smoke run, and 32-update stage pilot completed during setup.
Detached ongoing research is now managed by `ks3-codex-research.service`; see
[KS3_BACKGROUND.md](KS3_BACKGROUND.md) for its latest-run status and UTC-dated
ablation notes. While that worker is active, launch all experiments through
its `researchctl.py` registry to preserve exclusive GPU use and monitoring.

The primary candidate uses eleven active KS3 FFNs, including the parallel layer-8 FFN:
width 768, 1,536 rank-1 charts per FFN, GS32 W8A8 projections, FP16 nonlinearities,
FP32 projection masters and geometry, zero-initialized spine coefficients, learned
warps, coordinate bypass, and ReLU squared. There are **26,087,424 learned FFN parameters**.
This count is a recorded property, not a constraint; larger and smaller chart counts are supported.
The full 84,602,880-row n-gram table and the existing transformer topology remain present.

The primary configuration uses an overall learning-rate multiplier of 0.8 and geometry
Adam multiplier 0.125. The MI300 port's ANVIL schedules and BF16 spectral products remain;
KS3's FP32 projection masters use the same cautious update equations directly in FP32.
The other components retain the current MI300 precision and execution paths.

The measured pilot estimates initial LR/stage timing. `schedule_time_fraction` influences
that estimate; `schedule_base_steps` can set the initial schedule explicitly. These are
scheduling parameters, never termination conditions. Beyond the horizon, the final batch,
attention, full-vocabulary objective, LR floor (0.30), and clamped final rail momentum remain
valid for arbitrarily later steps. The same optimizer moments, sparse update counters,
input generator, candidate buffers, and asynchronous row prefetch remain live. Sparse Adam's
beta history grows without resetting prior events. No checkpoint/reload or data rewind occurs
at a schedule boundary; the original finite underlying shard stream is unchanged.

`first_validation_seconds` and `validation_interval_seconds` schedule repeated **full**
validation against complete elapsed wall time, at the next completed sparse-update cycle.
Missed deadlines during an evaluation are skipped to avoid consecutive catch-up evaluations.
Every evaluation uses the first 10,485,760 validation tokens and the control's final attention
windows and sequential YaRN scaling, then restores training settings. Before the initial
horizon, evaluations score the raw iterate. After it, the four original tail accumulators
continue at their regular EMA cadences. Evaluation temporarily ships their blend and restores
the exact raw weights afterward, including on exceptions; moments and accumulators are retained.
`weights` records which iterate was evaluated. No destructive tail shipping ends training.

`first_target_hit` and `time_to_target_seconds` record the first actual measured full target
hit, including evaluation/restoration time. `complete_wall_seconds` additionally includes all
remaining cleanup and any requested inference snapshot. No interpolated target times are used.

## Evaluation and timing

- Every trial score uses the same first **10,485,760 FineWeb validation tokens**, full-vocabulary
  loss, canonical mask, softcap, and final attention-window setting as the MI300 control.
  Sampled training objectives are logged separately as `train_objective`.
- Source data hashes must match the control. The original training and validation loaders are used;
  no retokenization, stream substitution, or small-prefix research score replaces the official metric.
- Complete-run timing starts immediately before launching `torchrun` and ends after it exits.
  It includes model setup, training, every validation, and any requested checkpoint write.
  Dataset hashing and disposable pilots run before this timer, as in the control.
- Actual complete-run wall time determines success. A pilot estimate, partial validation,
  interrupted run, or target hit after the control's duration cannot be labeled an exploratory win.
  Such trials still continue to measure their actual time to target.
- Launchers serialize KS3 jobs with a GPU lock. Run no other GPU workload during a measured trial.
  GPU information, resolved configuration, source archive, data hashes, token counts,
  parameter counts, validation points, and completion status are saved with every run.

`metrics.jsonl` contains validation loss against elapsed wall time. `result.json` records
the independent complete-run duration and the comparison outcome. `save_checkpoint: true`
writes the same complete inference snapshot as the MI300 runner, including the large n-gram
table; it is off by default, matching the control. This snapshot is not a resumable optimizer checkpoint.

The README's unmodified token-stream rule and restriction on extra Torch compilation flags
apply. No new `torch.compile` or Inductor flags are introduced. The existing MI300 hardware
adaptations are documented in `MI300.md`. A successful single trial here is an exploratory
result against the measured control. An official record still requires the README's ≤3.28
mean validation loss, p<0.01 evidence for an ML change, and faster timing on the same hardware.

## Change a candidate

Copy `configs/ks3/primary.json` and change its chart count, rank, batch-token schedule,
learning-rate multiplier, geometry rate, initial schedule timing, or wall-time validation cadence.
Chart count has no upper bound tied to the MLP parameter count. Memory and measured execution
time determine feasibility. The primary batch schedule is an initial choice, not a matching rule.

A new chart/rank/precision/batch configuration needs a new disposable pilot:

```bash
python3 ks3_research/ops/researchctl.py start --config configs/ks3/candidate.json \
  --label candidate-pilot --kind pilot --hypothesis 'Time the changed architecture or batch configuration.'
```

Set that candidate's `pilot_path` to the assigned run's `artifacts/pilot.json`, then launch it
through the registry as a trial. LR-only changes
can reuse the timing pilot. Repeat promising candidates across independent seeds before
making a convergence claim; the existing MLP control is a single seed.

The pilot jumps between schedule positions to cover all batch/attention stages, sampled
vocabulary sizes, sparse update cadence, twin rails, embedding split, tail averages, and
validation. Its losses are functional checks, not a continuous training experiment.

## Verification and source

```bash
CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' .venv-mi300/bin/python \
  -m unittest test_ks3_screening test_ks3_continuation test_ks3.TimePlanningTests -v
python3 ks3_research/ops/researchctl.py start --config configs/ks3/primary.json \
  --label continuation-check --kind smoke --hypothesis 'Verify real updates and evaluations across the schedule boundary.'
```

Verification retained under `logs/ks3-verification/`, `logs/ks3-pilot-primary/`, and
`logs/ks3-smoke-validation-resume/` covers fused outputs and all surrogate gradients,
the full-width topology, all eleven FFN replacements, shared initialization, a chart count
larger than the original FFN budget, time-based planning, and strict loss/time comparisons.
The original four-update smoke test evaluated midway and resumed training. The corrected
continuation smoke runs 36 actual updates across the embedding split and planned horizon,
including twelve beyond it, four explicitly partial diagnostic evaluations, forced sparse
history growth, and preserved input/prefetch/sample state assertions. It cannot claim a target hit.
CPU tests compare open prefetch against the finite reference across stages and evaluations,
check original optimizer/tail parity, exact weight and YaRN restoration, far-future schedules,
full-only target termination, and wall-time watchdog compatibility. Evidence lives under
`logs/ks3-continuation-verification/` and the registered smoke's artifacts.
The pilot found finite losses, gradients, and weights throughout all 32 updates;
all 101,376 coefficients and all 33,792 warp parameters moved. Peak allocation was 143.62 GiB.

The KS3 kernel and reference are vendored without code changes from
[`curvedinf/k-splanifolds` at `ea7a9c1`](https://github.com/curvedinf/k-splanifolds/tree/ea7a9c1121efe711345bd9e6cf22e69401c0c7b4).
Attribution, exact hashes, and the upstream CC-BY-SA-4.0 license are in `ks3_research/vendor/`.
The [report](https://github.com/curvedinf/k-splanifolds/blob/ea7a9c1121efe711345bd9e6cf22e69401c0c7b4/experiments/nanogpt_ks2/reports/2026-10-24-ks3-wallclock/REPORT.md)
provides the zero-spine, W8A8/FP32-master starting recipe and wall-clock experimental objective.
Its width-96 numerical results are not transferred to this model. Whole-model graph replay,
the report's local-attention kernels, and fused head loss are subsequent research options;
this first full-size trial uses the existing MI300 attention and loss implementations.
