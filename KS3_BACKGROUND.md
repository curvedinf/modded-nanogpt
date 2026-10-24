# Detached KS3 research

`ks3-codex-research.service` runs Codex **gpt-6-astra**, reasoning **max**, with
`--dangerously-bypass-approvals-and-sandbox` (YOLO). It is a system service, with
no terminal or SSH-session dependency. Its single saved task is
[`configs/ks3/autonomous_prompt.txt`](configs/ks3/autonomous_prompt.txt).
The supervisor resumes the same session and prompt if an exec turn ends, with
a delay rather than a busy restart loop. It is enabled at boot.

The fixed MLP control is 3.7936333656311034 full-validation CE in
2091.555386705 seconds. Research optimizes loss versus complete wall time, with
free parameter, batch, token and update budgets. The continuing target is
**3.793633 or lower, in progressively less time**. The MLP is not rerun.
Full trials continue until a measured full target hit, significant-regression cancellation,
real failure, or user stop. Schedule exhaustion and the control's runtime are not stop conditions.
Wall-time validation repeats while training continues through the initial schedule horizon.

## Independent monitoring

From `/root/modded-nanogpt`, without interacting with the background Codex:

```bash
python3 ks3_research/ops/researchctl.py status --brief
tail -n 20 logs/ks3-autoresearch/notes.log
```

Every experiment has a monotonic ID and its own systemd service under
`logs/ks3-autoresearch/runs/run-NNNNNN-label/`. `latest.json` and the `latest`
symlink deterministically identify the newest experiment; `latest_trial.json`
identifies the newest full trial even when the latest experiment is a pilot.
Use `status` without `--brief` for process, service, session and path details.

| Artifact under `logs/ks3-autoresearch/` | Purpose |
| --- | --- |
| `supervisor.json` | Codex PID, session ID, model, effort, heartbeat and event-log path |
| `registry.json`, `status.json` | Current run and machine-updated status |
| `latest/artifacts/metrics.jsonl` | Latest experiment's loss, wall time and training-token observations |
| `latest/pruning.jsonl` | Every early-stop assessment and its evidence |
| `latest/artifacts/result.json` | Completed timing and full-validation outcome, when available |
| `best.json` | Fastest completed full-validation target hit faster than the control; absent until achieved |
| `best_target.json` | Fastest completed target hit, including slower-than-control evidence |
| `ledger.jsonl` | Finished, cancelled and failed run outcomes |
| `notes.log`, `notes.jsonl` | Short UTC-dated ablation notes and decisions |
| `codex/attempt-*.jsonl` | Background Codex event history |

The start command requires a hypothesis and writes a UTC note. The watchdog
automatically writes an outcome note. The task prompt additionally requires
one to three sentences explaining the result and next direction per ablation.
Each run retains its own notes, configuration, source archive and logs.

## Monitoring cadence and cancellation

Codex is instructed to wait about two minutes between ordinary observations.
A separate run watchdog checks every second without model calls, and every
training update is logged. The candidate must first reach a loss strictly below
5.0; this milestone stays latched if loss later rises. Thereafter the first
compatible observation at least 0.01 worse than the dynamic current best causes
cancellation. The rule is one-sided: improvement never triggers cancellation.
The current best can be MLP or any recorded continuous KS3 trial; it excludes the
candidate itself and all pilot/smoke runs. The old 100k-token, fixed-MLP,
0.5/10%, two-observation rule is obsolete.

The control recorded only a final full validation. Consequently, this early
gate is explicitly a **training-curve heuristic**, with matching auxiliary-loss
phases, sampled-vocabulary sizes, prefix weights, MTP weights (tolerance 1e-6),
attention conditions and sampled-batch geometry. For each reference run, use its
latest compatible nonfuture measurement within 15 wall seconds, then the lowest
across runs. Never align by step or interpolate an unmeasured value. Incompatible,
future or stale training observations cannot establish a regression.

Full validation is compared separately against canonical full scores already
achieved by that elapsed time; a stopped model can retain its measured score.
Legacy intermediate evaluations with nonfinal attention settings are excluded.
The binding policy is `configs/ks3/screening_policy.json` and `AGENTS.md`.
Each run retains a policy snapshot, hashed reference-curve snapshots,
`screening_state.json` (including the latch witness), and `pruning.jsonl` with
selected reference run/time/objective, signature, gap and cancellation evidence.
`mlp_reference.json` supplies historical measurements only; its old embedded
pruning-policy fields are not used. Actual wins require the full 10,485,760-token
validation and measured complete runtime, as described in [KS3.md](KS3.md).

Cancel only the current experiment, allowing research to continue:

```bash
python3 ks3_research/ops/researchctl.py cancel --reason 'User requested this candidate stop'
```

Stop the research worker and request cancellation of its active experiment:

```bash
python3 ks3_research/ops/researchctl.py stop
```

This creates a persistent `STOP` marker. The worker is instructed never to
remove it or resume against an explicit stop. Disconnecting SSH does not stop
either the Codex service or experiment services.
