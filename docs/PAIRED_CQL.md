# CQL ablation: learned-policy versus recorded-action backups

This experiment asks whether CQL's performance changes when its critic learns
from recorded successor actions rather than actions proposed by an evolving
policy. A **Bellman backup** is the target used to train the critic: the observed
reward plus a discounted estimate of future value.

| Variant | Action evaluated at the next observation in the backup |
|---|---|
| Learned-policy backup (`--backup actor`) | An action sampled from the current learned policy |
| Recorded-action backup (`--backup sarsa`, called SARSA-CQL) | The successor action recorded in the dataset |

**Both variants train an actor and twin critics for 500k updates, and both deploy
the trained actor.** SARSA-CQL does not freeze the actor or replace it with a
dataset lookup at deployment. The intended intervention is only the successor
action used in the critic target; actor training and conservative regularisation
retain the same settings.

The comparison uses five **matched seed pairs** (123–127). Within each pair,
both runs start from identical network parameters and share the data, random-draw
schedule and evaluation winds. Their learned parameters can subsequently diverge.
These are purpose-built ablation runs, not a comparison of separately tuned
main-paper baselines.

To view the saved comparison without training, run
`.\review.cmd python scripts/reproduce_paper.py --figure 8` on Windows, or
`uv run --locked python scripts/reproduce_paper.py --figure 8` on Linux/macOS.
The backup comparison is panel (d); the instructions below are for rerunning it.

## Matched quantities

| Quantity | Both arms |
|---|---|
| Seeds / initial state | 123–127; identical actor, critic, target and initial dual tensors within each seed |
| Data | Same verified 24M-transition augmented replay and frozen normalisation |
| Replay stream | CPU generator seeded with `seed + 10000` |
| Learning draws | Torch reseeded at each update with `seed * 1000003 + step` |
| Actor entropy | Fixed temperature 0.1 |
| Bellman-target entropy | Disabled in **both** arms |
| Conservative loss | Base weight 1, adaptive target gap 30, initial dual multiplier 0.5 |
| Conservative samples | 10 each from uniform, current-policy and successor-policy proposals |
| Optimisers | Actor 1e-5, critics 3e-5, dual 1e-4; batch 256, discount .99, target rate .005 |
| Assessment | Same 20 validation winds, every 10k updates including zero |

SARSA still draws the unused successor actor action before replacing it with
the recorded successor action. This preserves the random-number schedule of
the conservative proposals. The **draw schedule** is matched, not the resulting
actions: actor distributions and adaptive conservative multipliers may diverge
after learning begins, as consequences of the changed backups.

The exact settings are in `configs/backup_actor.json` and
`configs/backup_sarsa.json`; the isolated loss is in `ablations/cql_backup.py`.
Do not use the main `scripts/train.py --algorithm cql` as the actor arm: its
historical RNG schedule differs from this matched ablation runner.

## Fresh reproduction

The default uv environment is CPU-only. For CUDA examples below, use the
separate GPU interpreter described in [SETUP.md](SETUP.md#optional-cuda-training)
instead of `uv run --locked python`.

Generate and verify the full replay as described in `REPRODUCING.md`. Use
`uv run --locked python scripts/cql_backup_commands.py --json` to obtain all training,
verification, evaluation and summary commands with explicit dependencies.
The generator prints a plan only: it never starts training or submits jobs.
There are 10 independent fits and 510 validation tasks, totalling 10,200
episodes. The plan has no held-out-test commands.

Any CUDA-capable GPU supported by the installed PyTorch build can be used, but
use the **same GPU model and runtime within each actor/SARSA pair**. The original
123/124 pairs used Quadro RTX 6000 and the 125–127 pairs used NVIDIA L40S;
different seed pairs need not share a GPU model. Runtime and code hashes are
written to `run.json`. `scripts/verify_cql_pairs.py` rejects mismatched pairs,
changed kernels, invalid checkpoint hashes, unequal initial states, unequal
replay/Torch streams or missing checkpoints. It does not require learned
networks or adaptive weights to remain identical.

Initial tensors are generated reproducibly from each seed. Historical initial
files are not bundled and are **not required for a fresh paired experiment**.
If retained externally, pass the same file to both runs using
`--initial-checkpoint initial_cql_s123.pt`. Only step-zero files with matching
seed, configuration, normalisation and empty optimiser histories are accepted.
The optional `scripts/verify_cql_initials.py` compares all five fresh CPU
initialisations with these historical files; the same-runtime five-seed check
is recorded in `provenance/cql_initialization_five_seeds.json`.

## Resume and verification scope

The runner refuses an existing output directory for a new run. To continue an
interrupted run, use `--resume` with its committed checkpoint and the **same**
backup, seed, data, output and numerical runtime. Model, optimiser, replay RNG
and global RNG states are restored; hashes link them to `run.json`. For example:

```sh
uv run --locked python scripts/train_cql_backup.py --backup actor --data-dir data/train --seed 123 --device cuda --output runs/cql-paired/actor/seed123 --resume runs/cql-paired/actor/seed123/checkpoints/step000300000.pt --steps 500000
```

The kernel parity check compared 1,000 updates of **each arm** against
the archived kernel on the full replay, including metrics, networks,
optimisers and RNG states. The all-seed initialization audit extends the
initial-state check; it is not a rerun of the ten historical 500k fits. Exact
CUDA trajectories can differ across software/hardware even with identical
seeds. New training/evaluation outputs stay separate from archived paper data.

The command-line workflow also passed a small end-to-end check for all
five pairs: two updates, saved-state verification, diagnostic validation and
summary/plot generation. This checks software wiring, not benchmark quality;
see `provenance/paired_cql_cli_smoke.json`.
