"""Build an allowlisted source archive; never include replay or run folders."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {'algorithms', 'ablations', 'configs', 'datasets', 'environments',
               'training', 'extraction', 'evaluation', 'demo', 'scripts', 'tests',
               'assets', 'results', 'provenance', 'docs', 'checkpoints', '.github'}
ROOT_FILES = {'README.md', 'NOTICE.md', 'LICENSE', 'CITATION.cff', 'pyproject.toml',
              'CONTRIBUTING.md',
              'requirements-reference.txt', '.gitignore', '.gitattributes',
              'core_pivot_truncated_20261019.py', 'run.py', '.python-version', 'uv.lock',
              'setup.cmd', 'review.cmd'}
SUFFIXES = {'.py', '.json', '.npz', '.png', '.gif', '.pdf', '.md', '.txt', '.toml', '.cff', '.csv',
            '.yml', '.yaml', '.sh', '.pbs', '.cmd', '.tex', '.sty', '.tikz'}


def release_files():
    selected = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if '__pycache__' in relative.parts or '.git' in relative.parts:
            continue
        if relative.as_posix() == 'data/README.md':
            selected.append(path)
        elif len(relative.parts) == 1 and path.name in ROOT_FILES:
            selected.append(path)
        elif relative.parts[0] in DIRECTORIES and (path.suffix in SUFFIXES or path.name == 'LICENSE'):
            if path.suffix in {'.pt', '.pth', '.npy', '.mp4'}:
                raise ValueError(f'Large runtime artifact in release: {relative}')
            selected.append(path)
    return selected


def audit(files):
    findings = []
    for path in files:
        if path.suffix in {'.py', '.sh', '.pbs'}:
            raw = path.read_bytes()
            if b'\r\n' in raw:
                findings.append(f'CRLF source: {path.relative_to(ROOT)}')
            text = raw.decode('utf-8')
            for forbidden in (() if path.name == 'package_release.py' else ('C:/Users/', 'C:\\Users\\', '/rds/general/user/', 'BEGIN OPENSSH PRIVATE KEY', 'BEGIN RSA PRIVATE KEY')):
                if forbidden in text:
                    findings.append(f'Non-portable/private source: {path.relative_to(ROOT)}')
    if findings:
        raise ValueError('\n'.join(findings))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not (ROOT / 'LICENSE').exists():
        p.error('Missing LICENSE: source packaging requires the repository licence.')
    files = release_files()
    audit(files)
    hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    manifest = dict(schema=1, license_status='selected',
                    files=hashes, file_count=len(files), uncompressed_bytes=sum(p.stat().st_size for p in files),
                    excluded=['full replay', 'model checkpoints', 'generated outputs', 'videos', '.git metadata', 'machine-specific files'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('xb') as stream, zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.writestr('CORE-PIVOT-release/' + path.relative_to(ROOT).as_posix(), path.read_bytes())
        archive.writestr('CORE-PIVOT-release/release_manifest.json', json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(path=str(args.out), files=len(files), bytes=args.out.stat().st_size,
                         sha256=hashlib.sha256(args.out.read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
