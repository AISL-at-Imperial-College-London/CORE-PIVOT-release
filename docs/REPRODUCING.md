# Reproducing the paper

Start with the locked CPU environment in the [README](../README.md).
For examples requesting CUDA, use the separate GPU environment in
[SETUP.md](SETUP.md#optional-cuda-training), replacing `uv run --locked python`
with that environment's Python path. The default uv environment is CPU-only.

Run commands from the repository root, in the installed environment. Output
directories must be new unless a command explicitly supports resume. A training
seed is not a wind seed. The five training seeds are **123–127**; the fixed PD
baseline is not trained five times.

## Evaluate released weights first (no training)

For an independent rollout of the paper's selected controllers, download the
private checkpoint asset and run the reviewer launcher:

```sh
uv run --locked python scripts/download_checkpoints.py
uv run --locked python scripts/download_checkpoints.py --verify-only
uv run --locked python run.py evaluate iql --all-seeds
uv run --locked python run.py evaluate core-pivot --all-seeds
uv run --locked python run.py evaluate all --all-seeds --split test --allow-test
```

This requires repository access, but no full replay or GPU. Validation is the
default; `--winds 1` is a two-episode software check. The selected-weight manifest
fixes model hashes, seed identities, training steps and BC dependencies. All
outputs are fresh evaluations, not copies of the archived paper statistics.
Use `--dry-run` to see a plan without executing it. Results are written under
`runs/evaluations/selected/`; choose `--output-dir` to keep software checks, validation,
test and robustness batches separate. See [checkpoint provenance](../checkpoints/README.md).

The following sections describe fresh training and lower-level controls. The
reviewer launcher uses `data/paper` and `runs/<family>_s<seed>` by default; the
explicit examples below use `data/train` and `runs/<family>/seed<seed>`. Both are
supported. Pass `--data-dir`/`--runs-dir` deliberately when organising your own
experiments rather than assuming that the two directory layouts are interchangeable.

## 1. Generate the historical dataset

```sh
uv run --locked python scripts/prepare_data.py --output data/train
uv run --locked python scripts/verify_data.py --data data/train
```

This generates 200 gain pairs × 160 ordinary winds × 750 transitions, in the
historical row order. It requires about **5.52 GB** for the seven replay arrays
plus temporary working space. Full-data generation checks the archived
reference hashes, not just array dimensions. See `data/README.md` and
`configs/dataset_split.json` for the exact observation, reward and split rules.

The nominal PD gains are (2, 1), with sampled perturbations. The best historical
controller is selected afterwards by mean training-wind cost; it is not the
centre of the gain perturbations. Stronger winds never enter training.

The prepared replay already includes the historical fixed previous-action
augmentation. **Do not add another layer of training noise.** `--clean` is a
different dataset, not the main-paper reproduction.

## 2. Fit the selected methods

All main learners receive 500k updates. Repeat each command for seeds 123–127,
changing both `--seed` and the output path. CPU works for tests; use
`--device cuda` for full fitting. No GPU model is hardcoded.

For the simpler dependency-aware workflow, use `uv run --locked python run.py train METHOD
--seed 123 --device cuda --data-dir data/train`. It automatically fits missing BC
dependencies, reuses matching completed runs and refuses to overwrite incomplete
runs. `easybcq` and `hoflon` share a single `qbeta` fit. Alternatively, the
following explicit commands give full control over output locations:

```sh
uv run --locked python scripts/train.py --algorithm bc --data-dir data/train --seed 123 --device cuda --output runs/bc/seed123
uv run --locked python scripts/train.py --algorithm qbeta --data-dir data/train --seed 123 --device cuda --output runs/qbeta/seed123
uv run --locked python scripts/train.py --algorithm td3-bc --data-dir data/train --seed 123 --device cuda --output runs/td3-bc/seed123
uv run --locked python scripts/train.py --algorithm iql --data-dir data/train --seed 123 --device cuda --output runs/iql/seed123
uv run --locked python scripts/train.py --algorithm cql --data-dir data/train --seed 123 --device cuda --output runs/cql/seed123
uv run --locked python scripts/train.py --algorithm core-pivot --data-dir data/train --seed 123 --device cuda --bc-checkpoint runs/bc/seed123/checkpoints/step000500000.pt --output runs/core-pivot/seed123
```

CORE-PIVOT uses **the same seed-123 BC500k model for every critic seed**.
EasyBCQ/HOFLON-EasyBCQ use seed-matched BC500k models at evaluation. Behavioural
Q fitting does not need BC proposals: its successor actions are recorded in the
dataset. Candidate generation is attached only at deployment.

`uv run --locked python scripts/paper_commands.py` prints all 30 selected-setting fits in
dependency order without running anything. `--json` emits explicit argument
lists and dependencies for your own scheduler. The ablations below are separate.

| Method | Selected setting | Common paper checkpoint |
|---|---|---:|
| BC | Truncated Gaussian, 2×1024 | 500k |
| EasyBCQ | M=100, behavioural Q | 480k |
| HOFLON-EasyBCQ | M=100, ρ=0.1, SLSQP40 | 310k |
| CORE-PIVOT | λ=0.01; 10 BC + 40 uniform + previous + zero | 360k |
| TD3-BC | α=0.5 | 290k |
| IQL | expectile=0.98, β=5, advantage cap=20 | 500k |
| CQL | adaptive target gap=30 | 330k |

The CQL actor has entropy temperature 0.1, but the Bellman target has **no
entropy term**. Both critic and actor schedules matter; these are not generic
upstream defaults. Exact configs live in `configs/train/`. Equal update counts
do not imply equal computation, and the axis excludes separate BC fitting.

Use those configs through the training entry point. Some preserved low-level
class defaults reflect earlier experiments; instantiating a learner with no
explicit configuration is not a shortcut to the selected paper setting.

### Interrupted runs and learning curves

The simple launcher intentionally does not resume silently. Resume a native run
using its last committed checkpoint and the **same** data, seed and output folder:

```sh
uv run --locked python scripts/train.py --algorithm iql --data-dir data/train --seed 123 --device cuda --output runs/iql/seed123 --resume runs/iql/seed123/checkpoints/step000100000.pt
```

The step above is an example: use an existing checkpoint and retain its `.json`
and `.state.pt` companions. CORE-PIVOT also requires the same `--bc-checkpoint`.
Downloaded historical weights are evaluation assets, not native resume bundles.

For a learning curve, evaluate each saved checkpoint separately using the
lower-level evaluator below, replacing `--checkpoint`, `--expected-step` and the
output filename. Learners save every 10k updates; BC saves every 25k. Never use
test evaluations to select a checkpoint. `run.py evaluate --source trained` uses
the paper's selected step, not an automatic new checkpoint search.

## 3. Validate, then evaluate the fixed selection

Evaluate learner checkpoints every 10k updates (BC every 25k). Validation has
15 ordinary and five stronger winds. Example commands for seed 123:

```sh
uv run --locked python scripts/evaluate.py --method easybcq --checkpoint runs/qbeta/seed123/checkpoints/step000480000.pt --bc-checkpoint runs/bc/seed123/checkpoints/step000500000.pt --expected-step 480000 --seed 123 --split validation --output runs/easybcq_s123_validation.json
uv run --locked python scripts/evaluate.py --method hoflon --checkpoint runs/qbeta/seed123/checkpoints/step000310000.pt --bc-checkpoint runs/bc/seed123/checkpoints/step000500000.pt --expected-step 310000 --seed 123 --radius 0.1 --split validation --output runs/hoflon_s123_validation.json
uv run --locked python scripts/evaluate.py --method core-pivot --checkpoint runs/core-pivot/seed123/checkpoints/step000360000.pt --expected-step 360000 --seed 123 --split test --allow-test --output runs/core_s123_test.json
uv run --locked python scripts/evaluate.py --method best-pd --split test --allow-test --output runs/pd_test.json
```

For other explicit policies use `--method td3-bc`, `iql`, `cql`, or `bc` and
the checkpoint in the table. Test access requires `--allow-test`; do not select
new hyperparameters from test outcomes. The reported historical selection was
an exploratory joint ordinary/stronger validation trade-off, with one common
checkpoint per method, not independent per-seed optimisation.

For robustness, keep the same checkpoint and use `--condition all`, or e.g.
`--condition reversing_3`. Add `--trajectories` to retain state/action/wind
records. This uses the 20 validation identifiers under altered forcing; it is
not an extension of the reserved test set.

Deployment uses fixed policies and the best-feasible SLSQP fallback: useful
feasible iterates can be retained at the iteration limit; predicted-Q improvement
is not a rollout guarantee. CORE-PIVOT resets candidate RNG to 9001 + wind seed;
the evaluator controls the other policies' RNGs as well. Use separate processes,
not Python threads, for parallel CORE-PIVOT evaluation because its legacy adapter
temporarily intercepts SciPy optimisation. Frozen-IQL extraction checkpoints are
checked against the fingerprint of their source critic bundle.

Average over winds **within each seed first**. Then compute the mean, sample
SD (`ddof=1`), minimum and maximum of those five seed means. The fixed PD
baseline's SD describes wind-to-wind variability, not training-seed variability.

## 4. Main ablations

### Refinement radius and CORE-PIVOT conservatism

At fixed Q310k and seed-matched BC500k, repeat the HOFLON validation command
with `--radius 0.05`, `0.1`, `0.25`, `1`, and `box`; use the same candidate
and iteration budgets. `--radius 0` gives EasyBCQ with identical candidate
draws/ties. The main radius panel uses seeds 123 and 124; subsequent additional
seeds are a follow-up, not retrospective hyperparameter selection.

For the coefficient-zero ablation, rerun CORE-PIVOT for seeds 123 and 124 with:

```sh
uv run --locked python scripts/train.py --algorithm core-pivot --data-dir data/train --seed 123 --device cuda --bc-checkpoint runs/bc/seed123/checkpoints/step000500000.pt --set conservative_weight=0 --output runs/core_lambda0/seed123
```

Do not change twin critics, backups, sampler or search settings. Evaluate every
10k checkpoint under both validation conditions.

Use the evaluation entry point for deployment: it installs the paper's
best-feasible, non-decreasing-value SLSQP fallback, including useful iterates
at the iteration limit. The preserved training kernels by themselves are
not a substitute for that deployment adapter.

### Frozen-IQL extraction

Freeze each seed's selected IQL500k target critics and value network. Fit fresh
actors, never update the critics, and use σ=1 for **both** extractors:

```sh
uv run --locked python scripts/export_iql.py --checkpoint runs/iql/seed123/checkpoints/step000500000.pt --out runs/iql/seed123/frozen.pt
uv run --locked python scripts/train_extractor.py --source runs/iql/seed123/frozen.pt --data data/train --kind awr --coefficient 5 --seed 123 --device cuda --out runs/awr/seed123
uv run --locked python scripts/train_extractor.py --source runs/iql/seed123/frozen.pt --data data/train --kind ddpg_bc --coefficient 1 --seed 123 --device cuda --out runs/ddpg_bc/seed123
```

Repeat for all five seeds. Defaults are 50k extraction updates and 1k
checkpoints, including step zero. AWR uses cap20; DDPG+BC uses a fixed Q scale
estimated on 65,536 training samples and likelihood behavioural regression.
Both deploy the distribution mean. This differs from resuming the original
IQL actor and from standard end-to-end TD3-BC training.

Evaluate each extraction checkpoint with its original frozen source bundle:

```sh
uv run --locked python scripts/evaluate.py --method awr --source-critic runs/iql/seed123/frozen.pt --checkpoint runs/awr/seed123/checkpoints/step050000.pt --seed 123 --split validation --output runs/awr_s123_step50k_validation.json
uv run --locked python scripts/evaluate.py --method ddpg-bc --source-critic runs/iql/seed123/frozen.pt --checkpoint runs/ddpg_bc/seed123/checkpoints/step050000.pt --seed 123 --split validation --output runs/ddpg_bc_s123_step50k_validation.json
```

### CQL versus recorded-action CQL

The Figure 8(d) comparison contains **five matched seed pairs**,
including the additional actor-backup controls for seeds 125–127. Its separate
runner preserves the paired experiment's per-update torch seeding and
matched initialisation. It does not replace the original selected CQL test or
robustness baseline. Run both arms for seeds 123–127:

```sh
uv run --locked python scripts/train_cql_backup.py --backup actor --data-dir data/train --seed 123 --device cuda --output runs/cql-paired/actor/seed123
uv run --locked python scripts/train_cql_backup.py --backup sarsa --data-dir data/train --seed 123 --device cuda --output runs/cql-paired/sarsa/seed123
```

Both use the same conservative weight/adaptive gap,
actor entropy term and entropy-free Bellman targets; only the successor action
used by the target changes. The actor draw is still consumed in the SARSA arm
so random-draw schedules remain matched. Keep each pair on the same GPU model
and software stack; no particular GPU model is required for a new pair.

Print the complete **10-fit, 510-validation-task** plan without launching jobs:

```sh
uv run --locked python scripts/cql_backup_commands.py --stage train
uv run --locked python scripts/cql_backup_commands.py --json
```

After training, audit saved initial tensors, settings, runtime and random
streams before evaluating the 51 checkpoints per arm/seed (0–500k):

```sh
uv run --locked python scripts/verify_cql_pairs.py --runs runs/cql-paired --output runs/cql-paired/pair_audit.json
uv run --locked python scripts/cql_backup_commands.py --stage evaluate
uv run --locked python scripts/summarize_evaluations.py --input runs/cql-paired/validation --output runs/cql-paired/validation_summary.json --plot
```

The middle command **prints** the evaluation commands; run them with your
scheduler only after the audit passes. These use the same 15 ordinary and five
stronger validation winds and produce mean/min–max learning curves as SVG,
PDF and PNG. No test evaluation is included. See
[CQL backup ablation guide](PAIRED_CQL.md) for what is matched, resume checks and the
distinction between a new reproduction and the archived GPU trajectories.

## 5. Tuning and figures

The explicit tuning grids are in `configs/experiments.json`. Use
`scripts/train.py --set KEY=JSON` to change one supported training parameter;
do not duplicate source files. Q/candidate/radius sweeps use frozen networks
where appropriate. The paper's tuning was targeted, not exhaustive or
compute-matched.

```sh
uv run --locked python scripts/reproduce_paper.py
uv run --locked python scripts/animate.py --preview
uv run --locked python scripts/animate.py --workers 2
```

The figure command regenerates the current manuscript figures and performance
tables from archived results, without training. See [the figure map](GUIDE.md#figures-and-tables)
for individual-figure commands and optional schematic-source compilation.
The full animation uses saved 50Hz paths
with smooth positional interpolation and explicit gust timing; it is a
demonstration, not a new evaluation. Its wind field repeats the same spatially
uniform measured acceleration at fixed arrow anchors. Video rendering is
optional and can take several minutes.

To summarise your own evaluations, place their JSON outputs in a dedicated
directory and run:

```sh
uv run --locked python scripts/summarize_evaluations.py --input runs/evaluations --output runs/summary.json --plot
```

This requires all five matched seeds by default and rejects duplicate or
unmatched wind groups and mixed training/search settings. Keep checkpoint
JSON sidecars and their parent `run.json` to verify native training seeds.
For released original weights, keep the adjacent selected-weight manifest;
its hashes link the historical exports to their recorded seed identities.
Summarise each hyperparameter-sweep setting in a separate directory.
`--allow-partial` is only for incomplete checks, including historical checkpoints
without verifiable seed metadata. New outputs
are not mixed with the bundled paper-result archive.

## Numerical reproducibility

Use `requirements-reference.txt` and the original runtime when investigating
small differences. SciPy versions and nonsmooth twin-minimum optimisation can
change closed-loop paths, even when checkpoint bytes match. Preserve runtime,
data, proposal and checkpoint hashes with every result. Keep independent evaluation
batches separate. See [verification](VERIFICATION.md) for exactly what has been checked.
