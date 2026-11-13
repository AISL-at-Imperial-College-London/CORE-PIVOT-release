"""Export a compact frozen-IQL bundle from a native training checkpoint."""
import argparse
import hashlib
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if state['step'] != 500000:
        raise ValueError('Select the paper IQL 500k checkpoint, not a new per-extractor critic.')
    modules = {name: state['modules'][name] for name in ('q_target', 'vf', 'actor')}
    bundle = dict(step=state['step'], mean=state['mean'], scale=state['scale'], modules=modules,
                  source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('xb') as stream:
        torch.save(bundle, stream)
    print(args.out)


if __name__ == '__main__':
    main()
