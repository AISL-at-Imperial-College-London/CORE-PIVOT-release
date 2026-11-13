"""Validate available run identity without guessing seeds from filenames."""
import hashlib
import json
from pathlib import Path, PureWindowsPath

import torch


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _seed(value, source):
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f'{source} seed must be an integer')
    return value


def _selected_manifest_identity(path, step):
    """Resolve only a contained path whose bytes match the selected manifest."""
    manifest_path = path.parent / 'manifest.json'
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict):
        raise ValueError('Selected checkpoint manifest must be an object')
    if 'checkpoints' not in manifest:
        return None  # Unrelated dataset manifests are not identity evidence.
    entries = manifest['checkpoints']
    if not isinstance(entries, list):
        raise ValueError('Selected checkpoint manifest checkpoints must be a list')
    root, target = path.parent.resolve(), path.resolve()
    matches = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get('path'), str):
            raise ValueError('Selected checkpoint manifest needs checkpoint paths')
        name = entry['path']
        relative = Path(name.replace('\\', '/'))
        # Reject drive-qualified paths on every host, including POSIX reviewers.
        if (not name or relative.is_absolute() or PureWindowsPath(name).drive
                or PureWindowsPath(name).root):
            raise ValueError('Selected checkpoint manifest path must be relative and contained')
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError('Selected checkpoint manifest path escapes its directory')
        if resolved == target:
            matches.append(entry)
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError('Selected checkpoint manifest contains duplicate checkpoint paths')
    entry = matches[0]
    digest = entry.get('sha256')
    if not isinstance(digest, str) or len(digest) != 64 or digest.lower() != file_sha(path):
        raise ValueError('Selected checkpoint manifest hash mismatch')
    selected_seed = _seed(entry.get('seed'), 'Selected checkpoint manifest')
    if selected_seed is None:
        raise ValueError('Selected checkpoint manifest seed is missing')
    selected_step = entry.get('step')
    if (not isinstance(selected_step, int) or isinstance(selected_step, bool)
            or selected_step < 0 or step != selected_step):
        raise ValueError('Checkpoint and selected manifest step disagree')
    return selected_seed


def checkpoint_identity(path, expected_seed):
    path = Path(path)
    record = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(record, dict):
        raise ValueError('Checkpoint must contain a mapping')
    state = record.get('learner', record.get('trainer', record))
    if not isinstance(state, dict) or not isinstance(state.get('config', {}), dict):
        raise ValueError('Checkpoint state and configuration must be mappings')
    config = dict(state.get('config', {}))
    internal_identity = record.get('identity', {})
    if not isinstance(internal_identity, dict):
        raise ValueError('Checkpoint identity must be a mapping')
    seeds = [_seed(value, 'Checkpoint') for value in
             (record.get('seed'), config.get('seed'), internal_identity.get('seed'))
             if value is not None]
    if len(set(seeds)) > 1:
        raise ValueError('Checkpoint internal seeds disagree')
    saved_seed = seeds[0] if seeds else None
    evidence = 'checkpoint' if saved_seed is not None else 'unavailable in historical export'
    sidecar, run_path = path.with_suffix('.json'), path.parent.parent / 'run.json'
    if sidecar.exists() and run_path.exists():
        proof = json.loads(sidecar.read_text())
        if 'model_sha256' in proof and 'run_sha256' in proof:
            if proof['model_sha256'] != file_sha(path) or proof['run_sha256'] != file_sha(run_path):
                raise ValueError('Checkpoint/run sidecar hash mismatch')
            identity = json.loads(run_path.read_text())['identity']
            if saved_seed is not None and saved_seed != identity['seed']:
                raise ValueError('Checkpoint and run seed disagree')
            if config != identity['config']:
                raise ValueError('Checkpoint and run settings disagree')
            saved_seed = identity['seed']
            evidence = 'hash-linked checkpoint sidecar and run.json'
    selected_seed = _selected_manifest_identity(path, state.get('step'))
    if selected_seed is not None:
        if saved_seed is not None and saved_seed != selected_seed:
            raise ValueError('Checkpoint and selected manifest seed disagree')
        saved_seed = selected_seed
        evidence = 'hash-linked selected checkpoint manifest.json'
    if saved_seed is not None and saved_seed != expected_seed:
        raise ValueError(f'Checkpoint seed {saved_seed} does not match --seed {expected_seed}')
    config.pop('seed', None)  # A training seed is not a hyperparameter difference.
    return dict(policy_config=config, checkpoint_seed=saved_seed,
                seed_verified=saved_seed is not None, seed_evidence=evidence)


def register_configuration(record, registered):
    """One setting per named method; use separate input folders for sweeps."""
    if 'policy_config' not in record:
        raise ValueError('Evaluation lacks policy settings; regenerate it with the current evaluator')
    configuration = dict(policy=record['policy_config'], search=record.get('search'))
    fingerprint = json.dumps(configuration, sort_keys=True, allow_nan=False)
    method = record['method']
    if method in registered and registered[method] != fingerprint:
        raise ValueError(f'{method}: mixed policy/search settings; summarise each sweep setting separately')
    registered[method] = fingerprint
    return configuration
