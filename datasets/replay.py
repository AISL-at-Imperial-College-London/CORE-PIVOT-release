"""A small fixed offline replay buffer shared by the learning algorithms."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


FIELDS = ("observations", "next_observations", "actions", "next_actions",
          "rewards", "terminals", "timeouts")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ReplayBuffer:
    """Unstandardized observations, fixed-bound normalized actions, raw rewards.

    Algorithms own preprocessing and objectives. ``next_actions`` are SARSA
    targets from the same episode, not actions from an adjacent trajectory.
    """
    arrays: dict
    source: dict

    def __post_init__(self):
        if set(self.arrays) != set(FIELDS):
            raise ValueError("Replay must contain the conventional transition fields and next_actions.")
        count = len(self.arrays["observations"])
        if not count or any(len(array) != count or array.ndim != 2 for array in self.arrays.values()):
            raise ValueError("Replay fields must be nonempty 2D arrays with matching lengths.")
        for name in ("rewards", "terminals", "timeouts"):
            if self.arrays[name].shape[1] != 1:
                raise ValueError(f"{name} must have shape (transitions, 1).")
        if any(self.arrays[name].dtype != np.bool_ for name in ("terminals", "timeouts")):
            raise ValueError("Terminal and timeout fields must be boolean.")
        for name in ('observations', 'next_observations', 'actions', 'next_actions', 'rewards'):
            expected = 25 if 'observations' in name else 1 if name == 'rewards' else 3
            if self.arrays[name].shape[1] != expected or self.arrays[name].dtype != np.float32:
                raise ValueError(f'{name} must be a float32 array with {expected} columns.')
            for start in range(0, count, 100_000):
                chunk = self.arrays[name][start:start + 100_000]
                if not np.isfinite(chunk).all():
                    raise ValueError(f'Non-finite {name}.')
                if name in ('actions', 'next_actions') and np.abs(chunk).max() > 1 + 1e-6:
                    raise ValueError('Replay actions must use fixed normalized bounds.')

    def __len__(self):
        return len(self.arrays["observations"])

    def sample(self, batch_size, generator, *, device="cpu"):
        """Uniform transition minibatches; a caller-owned CPU RNG makes runs repeatable."""
        if batch_size <= 0 or len(self) == 0:
            raise ValueError("A positive batch size and nonempty replay are required.")
        indices = torch.randint(len(self), (batch_size,), generator=generator).numpy()
        batch = {name: torch.from_numpy(values[indices]).to(device)
                 for name, values in self.arrays.items()}
        # A timeout is a collection boundary; only a true task terminal disables bootstrap.
        batch["bootstrap_mask"] = (~batch["terminals"]).float()
        return batch

    @classmethod
    def from_dataset(cls, dataset, *, cache_directory=None):
        """Preallocate once; an optional fresh directory stores disk-backed arrays.

        A failed preparation has no complete manifest and is not reusable. We
        never overwrite an existing cache or concatenate a second full copy.
        """
        metadata = dataset.metadata
        if metadata["action_coordinates"] != "normalized":
            raise ValueError("Load actions with action_coordinates='normalized'.")
        horizon = metadata["transitions_per_episode"]
        count = dataset.population_episodes * horizon
        if count <= 0:
            raise ValueError("No complete episodes were selected.")
        shapes = {"observations": (count, metadata["observation_size"]),
                  "next_observations": (count, metadata["observation_size"]),
                  "actions": (count, 3), "next_actions": (count, 3),
                  "rewards": (count, 1), "terminals": (count, 1), "timeouts": (count, 1)}
        directory = None if cache_directory is None else Path(cache_directory)
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=False)
        arrays = {}
        for name, shape in shapes.items():
            dtype = np.bool_ if name in ("terminals", "timeouts") else np.float32
            arrays[name] = (np.empty(shape, dtype=dtype) if directory is None else
                            np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+",
                                                      dtype=dtype, shape=shape))
        offset = 0
        identities = []
        for episode in dataset.iterate_episodes():
            transitions = episode.transitions()
            length = len(episode.actions)
            if length != horizon or offset + length > count:
                raise ValueError("Episode length/count differs from the selected metadata.")
            if not all(np.isfinite(transitions[key]).all() for key in
                       ("observations", "next_observations", "actions", "rewards")):
                raise ValueError("Non-finite transition values.")
            if np.abs(episode.actions).max() > 1 + 1e-6:
                raise ValueError("Normalized actions must lie in [-1, 1].")
            rows = slice(offset, offset + length)
            for name in shapes:
                if name == "next_actions":
                    arrays[name][rows] = 0
                    arrays[name][offset:offset + length - 1] = episode.actions[1:]
                    # An interior terminal (if present) may not point to a later action.
                    arrays[name][rows][episode.terminations] = 0
                else:
                    values = transitions[name]
                    arrays[name][rows] = values[:, None] if values.ndim == 1 else values
            identities.append(dict(wind_seed=int(episode.infos["disturbance_seed"]),
                                   controller_id=int(episode.infos["controller_id"])))
            offset += length
        if offset != count or identities != dataset.identities():
            raise ValueError("Loaded episodes differ from the selected group manifest.")
        source = dataset.source_manifest()
        if directory is not None:
            files = {}
            for name, values in arrays.items():
                values.flush()
                files[f"{name}.npy"] = _sha256(directory / f"{name}.npy")
            manifest = dict(version=1, complete=True, transitions=count,
                            source=source, files=files)
            (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return cls(arrays, source)

    @classmethod
    def open_cache(cls, directory, dataset, *, verify=True):
        """Reuse only a complete cache whose requested group/source identity matches."""
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("version") != 1 or manifest.get("complete") is not True:
            raise ValueError("Replay cache is incomplete or has an unsupported version.")
        if manifest["source"] != dataset.source_manifest():
            raise ValueError("Replay cache belongs to a different dataset selection/source.")
        arrays = {}
        for name, expected_hash in manifest["files"].items():
            if verify and _sha256(directory / name) != expected_hash:
                raise ValueError(f"Replay cache hash mismatch: {name}.")
            arrays[Path(name).stem] = np.load(directory / name, mmap_mode="r", allow_pickle=False)
        if any(len(array) != manifest["transitions"] for array in arrays.values()):
            raise ValueError("Replay cache field lengths differ.")
        return cls(arrays, manifest["source"])
