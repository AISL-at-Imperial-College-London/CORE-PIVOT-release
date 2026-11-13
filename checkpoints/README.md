# Selected paper checkpoints

The original main-paper weights are distributed separately from Git in the
[private selected-checkpoints-v1 release](https://github.com/EvripidouKris/CORE-PIVOT-release/releases/tag/selected-checkpoints-v1).
Repository access is required. The committed `selected/manifest.json` fixes the
method, seed, step, file size and SHA-256 for each checkpoint and its BC dependency.
The download contains 35 files (about 267 MiB compressed).

From the installed checkout:

```sh
uv run --locked python scripts/download_checkpoints.py
uv run --locked python scripts/download_checkpoints.py --verify-only
uv run --locked python run.py evaluate core-pivot --seed 123
```

No training dataset or GPU is needed. The downloader checks every file before
installation and refuses to overwrite a different existing checkpoint. It can
reuse an existing GitHub credential without printing it. If authentication is
unavailable, download the ZIP while signed into GitHub and import it:

```sh
uv run --locked python scripts/download_checkpoints.py --archive PATH_TO_ZIP
```

Do not commit credentials or paste tokens into command arguments. A token supplied
through `GH_TOKEN` or `GITHUB_TOKEN` needs access to this repository's contents.
The downloader also supports GitHub CLI and Git credential-manager sign-ins.

## What these weights represent

These are the selected historical weights, not newly trained substitutes. BC uses
500k; EasyBCQ uses Q480k; HOFLON–EasyBCQ uses Q310k; CORE-PIVOT uses 360k;
TD3-BC uses 290k; IQL uses 500k; original CQL uses 330k. All learned methods use
seeds 123–127. Candidate-based methods use seed-matched BC500k, and all CORE-PIVOT
critics embed the same seed-123 BC500k proposal. The fixed PD baseline requires
no checkpoint.

Of the 35 files, 34 match hashes retained in the original evaluation export.
The original export did not retain a hash for TD3-BC seed 126. That file was
recovered from the recorded HPC checkpoint location, checked for the selected
step/settings, and reproduced all 20 recorded validation costs within a maximum
relative difference of 3.6e-7. Its current hash is now pinned. This is behavioural
consistency evidence, not a claim of historical hash equality. The audit is in
[`provenance/selected_checkpoints.json`](../provenance/selected_checkpoints.json).

The files retain the original bytes, including any optimizer state and historical
metadata. They are supplied for **evaluation**, not as a complete resume bundle:
the original replay-sampler and global RNG states are not uniformly available.
Fresh training saves the native, hash-linked resume files described in
[the reproduction guide](../docs/REPRODUCING.md).

The main CQL baseline is deliberately distinct from the later paired CQL/SARSA-CQL
ablation. That ablation has its own training commands and archived results;
none of its weights silently replace the main CQL checkpoint.

Loading is CPU-only with `torch.load(..., weights_only=True)`. The evaluator
checks the training step and seed evidence. File identities and runtime versions
are recorded in each new evaluation. Cross-hardware and cross-SciPy results need
not be bitwise identical to the historical HPC runs.
