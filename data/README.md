# Training data

The full replay is generated locally, not checked into Git. Run:

```sh
uv run --locked python scripts/prepare_data.py --output data/train
uv run --locked python scripts/verify_data.py --data data/train
```

Full generation writes 24,000,000 transitions (200 PD controllers × 160
ordinary wind realisations × 750 transitions). The seven NPY files occupy
5,520,000,896 bytes. Keep additional disk space for checkpoints. Generation is
vectorised across controllers but preserves the historical ordering and
numerics. Only the training wind IDs are generated.

## Array contract

| Array | Shape | dtype / units |
|---|---|---|
| observations / next_observations | N×25 each | float32; physical features, not standardised |
| actions / next_actions | N×3 each | float32; normalised to [-1, 1] |
| rewards | N×1 | float32; negative next-state tracking cost |
| terminals / timeouts | N×1 each | bool; finite-horizon terminal, no external truncation |

Observation columns, in order:

1. Position error (3) and velocity error (3).
2. Reference velocity 0.2 s ahead (3) and 0.4 s ahead (3).
3. Previous **physical** action (3), followed by normalised time (1).
4. Causal previous-transition disturbance estimate (3).
5. Current reference velocity (3) and reference acceleration (3).

Physical control is `u=5*a`, with each component bounded by ±5 m/s².
Dynamics are explicit Euler at 0.02 s; each episode lasts 15 s. Reward uses
the **successor** state/reference: minus squared position error minus 0.2
times squared velocity error. The final transition is terminal; recorded
successor actions are shifted within each episode and set to zero at its end.
They never leak from the next controller or wind episode.

## Fixed augmentation and normalisation

The paper's prepared learner replay contains fixed Gaussian noise on the
previous-action channels. Its standard deviation is 0.01 times half the
observed physical-action range, with RNG seed 104852. Interior successor
features are aligned with the corresponding next current-state features.
Rewards, actions and successor-action targets are not changed. This is a
cached training view; no such noise is added to evaluation observations.

Population means and standard deviations are calculated over training
observations only, using the archived float64 100k-row chunk procedure and
a 1e-6 minimum scale. Networks store the resulting buffers as float32.
Never refit normalisation on validation, test or robustness trajectories.

`configs/pd_generation_catalog.csv` preserves the original serialized gains
and parser convention. `configs/dataset_split.json` defines every wind ID.
`configs/historical_replay_sha256.json` commits the full historical arrays.
The generator stops if a full paper replay differs from those hashes.

`--smoke` creates six episodes (4,500 transitions) for software tests only.
`--clean` disables augmentation and therefore describes a different dataset.
Neither option reproduces the paper's fitting results.
