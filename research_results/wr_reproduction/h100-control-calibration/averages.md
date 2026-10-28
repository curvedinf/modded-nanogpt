# PR393 local control calibration

Frozen upstream commit: `426ab710eb0ff4d283d55ecc3391fb5e884e8d61`.

7 fresh independent seeds, with unchanged model/training sources, 1,053 updates per run and full 10,485,760-token final WR validation. The accepted local reference combines these with the preserved original, for 8 measurements. The fresh runs are also summarized separately. The user stopped the originally planned eleven-run batch and accepted its current average.

| Population | Mean time (s) | Time SD (s) | Mean validation loss | Loss SD |
| --- | ---: | ---: | ---: | ---: |
| 7 fresh runs | 31.076429 | 0.015285 | 3.2782429 | 0.0010326 |
| Original + 7 fresh runs | 31.079250 | 0.016246 | 3.2780250 | 0.0011374 |

SD is the sample standard deviation across runs. Original logs round time to milliseconds and loss to four decimal places; averaging does not restore the discarded precision.

| Run | Seed | Time (s) | Validation loss |
| --- | ---: | ---: | ---: |
| Original `20261028T180502Z` | Random default | 31.099 | 3.2765 |
| [1](runs/20261028T224939Z/result.json) | 601 | 31.054 | 3.2794 |
| [2](runs/20261028T225333Z/result.json) | 602 | 31.067 | 3.2793 |
| [3](runs/20261028T225636Z/result.json) | 603 | 31.102 | 3.2782 |
| [4](runs/20261028T225939Z/result.json) | 604 | 31.079 | 3.2784 |
| [5](runs/20261028T230241Z/result.json) | 605 | 31.084 | 3.2786 |
| [6](runs/20261028T230545Z/result.json) | 606 | 31.068 | 3.2766 |
| [7](runs/20261028T230848Z/result.json) | 607 | 31.081 | 3.2772 |

Every fresh result has a source manifest, full console log, environment capture and replay archive in this directory. No run was selected or pruned based on its loss or timing.
