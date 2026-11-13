# Reviewer walkthrough

**Easy to verify, possible to reproduce.** Start with saved results and released
controllers. Full dataset generation and training are optional.

Commands below are for Windows PowerShell, from the repository folder. Run one
line at a time. On Linux/macOS, follow [SETUP.md](SETUP.md), then replace
`.\review.cmd` with `uv run --locked python run.py`, or `.\review.cmd python`
with `uv run --locked python` for scripts. No notebook is required.

## 1. Set up once

Clone using the [README](../README.md), or update an existing clone:

```powershell
git pull --ff-only
.\setup.cmd
.\review.cmd --help
```

Expect `Environment imports OK` and `Setup complete. No training was launched.`
Do not activate a virtual environment or run the bare `python` command. For
installation/access problems, see [SETUP.md](SETUP.md).

## 2. Verify a released controller

```powershell
.\review.cmd python scripts/download_checkpoints.py
.\review.cmd python scripts/download_checkpoints.py --verify-only
.\review.cmd evaluate iql --seed 123 --winds 1 --output runs/reviewer/iql_smoke.json
```

The downloader verifies 35 checkpoint files. The two-episode check should print
approximately:

```text
validation_ordinary wind 51020: 69.9878
validation_stronger wind 51139: 237.872
```

The JSON is saved to `runs/reviewer/iql_smoke.json`. Small numerical differences
are possible across platforms. This verifies loading and inference, not all paper
results. Repeated evaluations refuse to overwrite outputs: choose a new filename.
Private checkpoints require GitHub access; see [checkpoint help](../checkpoints/README.md).

For a check without downloads:

```powershell
.\review.cmd evaluate best-pd --winds 1 --output runs/reviewer/pd_smoke.json
```

## Figures and tables

Generate the current paper's figures and performance tables, without training,
checkpoint downloads or the full replay:

```powershell
.\review.cmd python scripts/reproduce_paper.py
explorer .\figures\reproduced
```

Open `figures/reproduced/index.html` for a numbered gallery with captions and PDF links.

To inspect the map or reproduce a single figure:

```powershell
.\review.cmd python scripts/reproduce_paper.py --list
.\review.cmd python scripts/reproduce_paper.py --figure 10
```

Figure numbers below follow the manuscript revision dated 13 November 2026.
Numerical plots have SVG, PDF and PNG versions. Tables are exported as numeric
CSVs. The computational-time table is not included.

| Figure | Content |
|---|---|
| 1–3 | CORE-PIVOT workflow, action-search schematic and tracking-task schematic |
| 4–5 | Separate PD-gain and training-cost diversity plots |
| 6 | Validation learning curves |
| 7 | Disturbance profiles and robustness performance |
| 8 | Radius, conservatism, frozen-IQL extraction and paired CQL backup ablations |
| 9 | Historical state–action density |
| 10 | Stacked ordinary/stronger held-out test violins |
| 11 | Full-range learning curves |
| 12–15 | Shared-scale hyperparameter sweeps: ordinary/stronger, full/zoomed |
| 16–19 | Held-out, random-walk, step-gust and reversing-gust trajectory grids |

These commands read **archived data**, not freshly evaluated controllers. The
numerical figures are rendered from those archives; fonts and layout can differ
slightly across platforms. Schematic PDFs reproduce the current artwork without
requiring TeX. To rebuild those PDFs from their standalone TikZ sources, install
a TeX distribution and add `--build-schematics` to the command above. Known
superseded generated filenames are removed only after their replacements exist.

For tables alone, run `.\review.cmd python scripts/export_tables.py`. Outputs are
`held_out_test.csv`, `robustness_analysis.csv`, `table1_selected.csv` and
`trajectory_batch_summary.csv` (the separate trajectory-recording batch). The
selected-intensity main table uses $A=1.5,2.5,4$; full robustness values remain
available. Architecture and training settings are in `configs/train/` and
`configs/experiments.json`. To preview the drone animation, run
`.\review.cmd python scripts/animate.py --preview`.
CSV rows include means, sample SDs and ranges where available, with the statistical
population labelled. Blank fixed-PD robustness dispersion fields mean those
statistics are not retained in the compact archive—not zero variability.

## 3. Collect fresh results without retraining

Choose the evaluation you want; these can take longer than the two-episode check:

```powershell
.\review.cmd evaluate all --all-seeds --split test --allow-test --output-dir runs/reviewer/test
.\review.cmd python scripts/summarize_evaluations.py --input runs/reviewer/test --output runs/reviewer/test_summary.json
```

For robustness, keep the same selected checkpoints:

```powershell
.\review.cmd evaluate all --all-seeds --condition all --output-dir runs/reviewer/robustness
.\review.cmd python scripts/summarize_evaluations.py --input runs/reviewer/robustness --output runs/reviewer/robustness_summary.json
```

Held-out tests use 15 ordinary and five stronger winds. Robustness uses 17 conditions
with 20 matched winds each. The released selection covers seven learned controllers
and the fixed PD baseline. Ablation-specific policies such as SARSA-CQL require the
separate reproduction recipes below; they are not additional released main checkpoints.

Summaries average winds within each seed, then report mean, sample SD and min–max
across five seed means. The fixed PD varies across winds, not training seeds.
Fresh summaries are JSON; `reproduce_paper.py` always plots the archived paper results,
not these new files. Do not choose new settings from held-out test results.

## 4. Train one algorithm (optional)

**The following training command starts a 500k-update fit.** First generate the
full dataset once and verify it:

```powershell
.\review.cmd python scripts/prepare_data.py --output data/paper
.\review.cmd python scripts/verify_data.py --data data/paper
.\review.cmd train iql --seed 123 --dry-run
```

The dry-run prints the plan only. When ready:

```powershell
.\review.cmd train iql --seed 123 --device cpu
.\review.cmd evaluate iql --seed 123 --source trained
```

Replace `iql` with `bc`, `easybcq`, `hoflon`, `core-pivot`, `td3-bc` or `cql`.
Use `--all-seeds` instead of `--seed 123` for five fits. The launcher handles required
BC fits; EasyBCQ and HOFLON share a behavioural critic. IQL checkpoints are saved
under `runs/iql_s123/`. CPU training can be slow; for NVIDIA training, follow
[CUDA setup](SETUP.md#optional-cuda-training) first—the default environment is CPU-only.

For learning curves, resumes and new ablations, follow [REPRODUCING.md](REPRODUCING.md).
Radius changes need evaluation only; conservative-loss and backup changes need
critic retraining; frozen-IQL extraction trains new actors while keeping critics
fixed. The [CQL backup ablation guide](PAIRED_CQL.md) explains the learned-policy
versus recorded-action comparison.

## Storage and hardware

| Activity | Data storage, excluding Python and caches |
|---|---|
| View/redraw figures and tables | About 95 MiB numerical archives, mostly saved trajectories, plus schematic/demo assets; generated files extra |
| Evaluate saved controllers | About 267 MiB checkpoint download and 311 MiB extracted; allow both if retaining the ZIP |
| Check dataset generation only | About 1 MB replay arrays for 4,500 transitions |
| Reproduce full training | **5,520,000,896 bytes** of replay: 5.52 GB / 5.14 GiB, plus models, resume states and working space |

The full replay has 24 million transitions: current and next 25D observations
alone use 4.8 GB. **It is unnecessary for plotting or checkpoint evaluation.**
Python, PyTorch, uv's cache and outputs require additional disk space. Retaining
checkpoints across many seeds/ablations can exceed the replay size; inspect free
space before a campaign. No fixed total or portable runtime is promised.

To check data generation cheaply, without fitting a model:

```powershell
.\review.cmd python scripts/prepare_data.py --smoke --output data/smoke
.\review.cmd python scripts/verify_data.py --data data/smoke
```

The smoke replay is **not** suitable for reproducing the paper's training results.
Evaluation runs on CPU. Hardware/version changes can affect numerical search and
training outcomes; see [verified scope](VERIFICATION.md) and `requirements-reference.txt`.

## Inspect the implementation

Open the folder in any editor. Start with `run.py`, then `algorithms/` and
`configs/`; `environments/` defines the task, `evaluation/` implements deployment,
and `extraction/` contains frozen-IQL experiments. No source edits are needed for
the commands above. Run scripts through the terminal, not an editor's default
Python interpreter, to use the intended environment.
