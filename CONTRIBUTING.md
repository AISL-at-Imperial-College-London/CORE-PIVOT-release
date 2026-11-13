# Working on this release

Keep numerical changes separate from packaging or presentation changes.
Before changing a learner, add a regression test showing the intended change;
do not silently update a source hash to make a failing parity test pass.
Record configuration, replay identity, normalisation, seeds and runtime with
new experiments. Preserve train/validation/test separation.

Run `uv run --locked python -m unittest discover -s tests -v` before proposing a
numerical change. The full suite includes small optimiser-update tests; it is
separate from the no-training reviewer checks.
Large training runs are opt-in. Use a smoke dataset to check software, and
never label smoke results as benchmark performance. Tests and source scripts
use LF line endings (Windows `.cmd` launchers use CRLF); do not regenerate source
manifests from CRLF-transformed files without recording that change.

Historical main-paper results are immutable inputs in `results/paper_data/`.
Store new outputs under `runs/` and compare them explicitly. Do not replace
historical exports, pool separate rollout batches, or tune on held-out tests.

The archive builder excludes full replay arrays, model checkpoints, generated
videos and machine-specific files. Preserve the MIT license and the
separate third-party licenses and attribution described in `NOTICE.md`.
