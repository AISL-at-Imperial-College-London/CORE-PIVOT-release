"""Regenerate train-only replay: python scripts/prepare_data.py --output data/train."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.generation import generate

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='2 winds x3 controllers, not paper results')
    parser.add_argument('--clean', action='store_true', help='Disable historical fixed previous-action noise')
    args = parser.parse_args()
    manifest = generate(args.output, smoke=args.smoke, clean=args.clean)
    print(json.dumps({k: manifest[k] for k in ('complete', 'rows', 'episodes', 'smoke')}, indent=2))

if __name__ == '__main__':
    main()
