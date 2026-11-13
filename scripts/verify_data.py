"""Check a generated replay against both its manifest and historical fingerprints."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.generation import CONFIGS, file_hash

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data / 'manifest.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('Replay has no completion commitment.')
    for name, digest in manifest['files'].items():
        path = (args.data / name).resolve()
        if not path.is_relative_to(args.data.resolve()) or file_hash(path) != digest:
            raise ValueError(f'Changed replay file: {name}')
    historical = None
    if manifest['rows'] == 24000000 and manifest['previous_action_augmentation'].get('std_data_scaled') == .01:
        expected = json.loads((CONFIGS / 'historical_replay_sha256.json').read_text())
        historical = {name: manifest['files'][name] == digest for name, digest in expected['files'].items()}
        if not all(historical.values()):
            raise ValueError(f'Historical replay mismatch: {historical}')
    print(json.dumps(dict(passed=True, rows=manifest['rows'], historical_replay_sha256_match=historical), indent=2))

if __name__ == '__main__':
    main()
