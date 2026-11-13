# Verification and limits

This package preserves the experiment implementations and archived results. It
does **not** claim an independent rerun of every 500k-update experiment.
Detailed machine-readable evidence remains in `provenance/`.

## Data and numerical implementation

| Check | Verified scope | Evidence |
|---|---|---|
| Full replay regeneration | All seven 24M-transition arrays match historical SHA-256 hashes, including fixed augmentation and normalisation | `provenance/data_regeneration_audit.json` |
| Main training kernels | All six learner families match archived kernels for 1,000 updates at seed 123; resume reproduces update 1,001 | `provenance/training_parity_1000.json` |
| Frozen-IQL extraction | AWR and DDPG+BC match 1,000 updates; frozen values remain unchanged | `provenance/extraction_parity_1000.json` |
| CQL backup implementations | Both backup variants match 1,000 updates with shared archived initialisation and RNG schedule | `provenance/cql_backup_1000step_parity.json` |
| Deployment | First 20 actions match the archived CPU evaluator for six improvement methods in both wind conditions; all 340 robustness wind traces match | `provenance/evaluation_parity.json` |
| Paired CQL results | Five paired seeds, 51 checkpoints per arm and 10,200 archived validation episodes; paired initial state/settings/runtime verified | `provenance/paired_cql_results.json` |

These are same-runtime parity checks, not cross-platform determinism guarantees.
The reference used Python 3.12.14, PyTorch 2.9.1+cu130, NumPy 2.5.1 and SciPy 1.18.0;
historical HPC records used Python 3.12.3. New runs record their own runtime.

## Released checkpoints

All 35 selected files were identity-checked and loaded, with repeatable five-step
actions and no parameter updates. Thirty-four match retained historical hashes.
TD3-BC seed 126 had no archived hash: its recovered 290k export has the expected
settings and reproduces all 20 recorded validation costs (maximum relative error
3.6e-7). Its current pinned hash is not presented as a historical hash match.
See `provenance/selected_checkpoints.json`.

The eight main methods completed two-episode validation software checks in a clean
installation. These checks are not new benchmark estimates. Released weights are
evaluation assets, not full native replay/RNG resume bundles.

## Reviewer setup

Windows setup was tested with uv 0.13.7, downloaded Python 3.12.16 and locked CPU
PyTorch 2.9.1. Separate source copies, paths containing spaces, absent Python/uv
PATH entries, missing WinGet links and package-only uv installations were tested.
Setup/imports, launcher dry-runs, identity/download tests, evaluation and plotting
passed. The IQL seed-123 two-episode costs were 69.9878 and 237.872; a separate
reviewer laptop with Python 3.12.3 reported the same rounded costs.

No training was launched in these setup checks. Transient Windows file-access
warnings required retries during the initial installation. The WinGet installer
branch was not independently reinstalled by the audit. Linux/macOS/CUDA were not
locally verified by this setup audit; a manual multi-platform CI workflow is
provided, without claiming it has already passed remotely.

## Figures and release scope

Plotting reads hash-verified archives. Held-out statistics cover 720 episodes:
600 improvement-method, 100 BC and 20 fixed-PD episodes. Robustness and ablation
archives retain their own identities. Saved demo trajectories and interpolation
are for presentation; they do not alter performance statistics.

The five-seed paired CQL ablation does not replace the selected main CQL baseline.
The figure entry point targets the manuscript revision dated 13 November 2026,
including its separate dataset figures, stacked violins, tuning sweeps and
trajectory grids. Numerical figures are regenerated from archived arrays;
rendering can differ slightly across platforms. Schematic PDFs are bundled with
standalone TikZ sources, so viewing them does not require a TeX installation.

On 13 November 2026, the complete 19-figure command and performance-table exports
were run locally from the archived data. The Windows reviewer wrapper and optional
TikZ source rebuild were also checked. Figure/archive, table, launcher, packaging
and existing paired-CQL tests passed. These checks did not train any models or
rerun the controller evaluations; they verify figure reproduction and packaging.

Source packaging excludes full replay, checkpoint binaries, generated videos,
computational-timing records and machine-specific files. Repository and checkpoint
access remain private. Software authorship, paper authorship and third-party
licences remain distinct; see `NOTICE.md` and `CITATION.cff`.
