# KS3 research: latest user instructions

The user explicitly authorized resuming research at 2026-10-24T07:28:15.200728+00:00. Resume the detached manager now, implement and test the corrected pruning rule below, then launch meaningful optimization ablations. The earlier stop is revoked. Honor any NEW stop instruction or STOP marker.

For all future optimization ablations, follow `configs/ks3/screening_policy.json`:
- Judge loss versus elapsed wall time, never matched steps.
- Always let a KS3 candidate first reach loss strictly below 5.0 before comparative pruning (errors and explicit stop are exceptions).
- Thereafter stop on the first comparable observation where KS3 loss is **at least 0.01 worse** than the **current best**, whether that reference is MLP or a KS3 run. Use compatible loss definitions and evaluation/objective conditions at comparable wall time.
- This is one-sided: a better-performing run continues. Do not cancel for an absolute difference in either direction. Do not require two bad observations.
- Do not repeat an unchanged recipe merely to consume unused time or establish the remaining gap. Run meaningful optimization ablations and screen inferior candidates early. Do not force every ablation to run the whole length or reach the final target before pruning.
- Before resuming experiments, implement and test this corrected dynamic-reference pruning rule. The old fixed-MLP 0.5/10% two-observation watchdog is obsolete and must not be used.
- Keep short UTC-dated hypothesis/result/decision notes per ablation and the independent run registry.

These instructions supersede conflicting earlier prompts or research notes. The MLP control remains immutable and must not be rerun.

## New control generation authorized by user
Build full PR393 MoE (8 top1 experts, CPLM and token normalization), with verified transferable MI300 optimizations. The new KS3 architecture must retain the same MoE router/topology and use KS3 expert blocks. Preserve the historical MLP control. Short new-control tuning tests have a cumulative 1800-second runtime limit; freeze a recipe afterward and collect three independent full-validation controls to <=3.793633. Keep manager stopped until those controls complete, then resume indefinite KS3 optimization against the new control curves with the below5/one-sided+.01 rule. This explicitly authorizes new control runs; the old prohibition concerns rerunning the historical control unchanged.

## Latest user stop and target correction
Research is STOPPED by explicit user instruction. Do not resume the manager or GPU runs without new authorization. The new comparison target is loss 3.68 in about 451 seconds, not the historical 3.793633 threshold. See configs/control393/objective_correction.json for exact measured control values and the correction. Previous KS3 target hits at 3.793633 do not establish matched-loss wins. Correct active runner/config targets before any future authorized resume; preserve immutable original measurements.

## Latest user resumption
The user now explicitly authorizes restarting correctly. Resume MoE KS3 research with full-validation target<=3.68 and reference451.581954s (measured control CE3.683013308). This revokes the preceding stop. Do not count3.793633 target hits as wins. Preserve all earlier evidence and honor any subsequent stop.
