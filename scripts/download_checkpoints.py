"""Fetch or verify the selected paper checkpoints, without importing ML libraries.

Authentication is read from GITHUB_TOKEN, GH_TOKEN, ``gh auth token``, or the
configured Git credential helper, in that order. Credentials stay in memory.
For an offline transfer, pass ``--archive selected-checkpoints-v1.zip``.
The ZIP must contain exactly the unique paths named by the manifest, relative
to its directory, without a top-level folder or explicit directory entries.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import BinaryIO
import urllib.error
import urllib.parse
import urllib.request
import zipfile


DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / 'checkpoints/selected/manifest.json'
CHUNK_BYTES = 1024 * 1024
MAX_ARCHIVE_BYTES = 16 * 1024**3
MAX_EXTRACTED_BYTES = 32 * 1024**3
MAX_METADATA_BYTES = 2 * 1024**2
MAX_FILES = 10000
TIMEOUT_SECONDS = 60
_SHA256 = re.compile(r'[0-9a-fA-F]{64}')
_REPOSITORY = re.compile(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+')
_RESERVED_NAMES = re.compile(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', re.I)


class CheckpointError(Exception):
    """A safe-to-display checkpoint validation or download failure."""


@dataclass(frozen=True)
class FileSpec:
    path: str
    sha256: str
    size_bytes: int | None = None


@dataclass(frozen=True)
class Manifest:
    repository: str
    release_tag: str
    asset_name: str
    asset_sha256: str
    asset_size_bytes: int
    files: tuple[FileSpec, ...]
    checkpoint_count: int
    source_bytes: bytes


def _safe_path(value: object) -> str:
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise CheckpointError('Manifest/archive paths must be relative POSIX paths.')
    parts = value.split('/')
    if any(not part or part in ('.', '..') or part[-1:] in (' ', '.')
           or any(ord(char) < 32 for char in part) or _RESERVED_NAMES.fullmatch(part)
           for part in parts):
        raise CheckpointError(f'Unsafe manifest/archive path: {value!r}')
    if PurePosixPath(value).is_absolute():
        raise CheckpointError(f'Absolute manifest/archive path: {value!r}')
    return value


def _digest(value: object) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise CheckpointError('Every file and release asset requires a SHA-256 digest.')
    return value.lower()


def _size(value: object, *, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    if type(value) is not int or value < 0 or value > MAX_EXTRACTED_BYTES:
        raise CheckpointError('Invalid or excessive size_bytes in manifest.')
    return value


def load_manifest(path: Path) -> Manifest:
    try:
        with path.open('rb') as stream:
            raw = stream.read(MAX_METADATA_BYTES + 1)
        if len(raw) > MAX_METADATA_BYTES:
            raise CheckpointError('Checkpoint manifest exceeds the metadata size limit.')
        data = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise CheckpointError(f'Cannot read checkpoint manifest: {path}') from exc
    if not isinstance(data, dict) or type(data.get('schema_version')) is not int or data['schema_version'] != 1:
        raise CheckpointError('Expected checkpoint manifest schema_version 1.')
    repository = data.get('repository')
    if not isinstance(repository, str) or not _REPOSITORY.fullmatch(repository):
        raise CheckpointError('Manifest repository must be an owner/repository name.')
    tag = data.get('release_tag')
    if not isinstance(tag, str) or not tag or any(ord(char) < 32 for char in tag):
        raise CheckpointError('Manifest requires a release_tag.')
    asset = data.get('asset')
    if not isinstance(asset, dict):
        raise CheckpointError('Manifest requires an asset descriptor.')
    asset_name = _safe_path(asset.get('name'))
    if '/' in asset_name:
        raise CheckpointError('Release asset name must be a filename.')
    asset_size = _size(asset.get('size_bytes'))
    if not asset_size or asset_size > MAX_ARCHIVE_BYTES:
        raise CheckpointError('Release asset size is empty or exceeds the download limit.')
    checkpoints = data.get('checkpoints')
    if not isinstance(checkpoints, list) or not checkpoints or len(checkpoints) > MAX_FILES:
        raise CheckpointError('Manifest requires a bounded, nonempty checkpoints list.')
    files: dict[str, FileSpec] = {}
    folded: dict[str, str] = {}
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('method'), str) or not checkpoint['method']:
            raise CheckpointError('Every checkpoint requires a method name.')
        descriptors = [checkpoint]
        if 'bc_checkpoint' in checkpoint:
            descriptors.append(checkpoint['bc_checkpoint'])
        for item in descriptors:
            if not isinstance(item, dict):
                raise CheckpointError('Invalid checkpoint descriptor.')
            if any(type(item.get(key)) is not int or item[key] < 0 for key in ('seed', 'step')):
                raise CheckpointError('Every checkpoint requires nonnegative integer seed and step.')
            name = _safe_path(item.get('path'))
            spec = FileSpec(name, _digest(item.get('sha256')), _size(item.get('size_bytes'), optional=True))
            previous = files.get(name)
            if previous and (previous.sha256 != spec.sha256
                             or (previous.size_bytes is not None and spec.size_bytes is not None
                                 and previous.size_bytes != spec.size_bytes)):
                raise CheckpointError(f'Conflicting descriptors for checkpoint: {name}')
            if name.casefold() in folded and folded[name.casefold()] != name:
                raise CheckpointError(f'Case-colliding checkpoint paths: {name}')
            folded[name.casefold()] = name
            if previous is None or spec.size_bytes is not None:
                files[name] = spec
    if len(files) > MAX_FILES:
        raise CheckpointError('Manifest exceeds the file-count limit.')
    for name in files:
        if any(str(parent).casefold() in folded for parent in PurePosixPath(name).parents if str(parent) != '.'):
            raise CheckpointError(f'Checkpoint path conflicts with a parent directory: {name}')
    if sum(item.size_bytes or 0 for item in files.values()) > MAX_EXTRACTED_BYTES:
        raise CheckpointError('Manifest exceeds the extracted size limit.')
    return Manifest(repository, tag, asset_name, _digest(asset.get('sha256')),
                    asset_size, tuple(files.values()), len(checkpoints), raw)


def _stream_digest(stream: BinaryIO, *, limit: int, destination: BinaryIO | None = None) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = stream.read(min(CHUNK_BYTES, limit - size + 1))
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise CheckpointError('File or download exceeds its declared size/stream limit.')
        digest.update(chunk)
        if destination is not None:
            destination.write(chunk)
    return digest.hexdigest(), size


def _is_link(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0)
                                            & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400))


def _destination(output: Path, name: str) -> Path:
    destination = output.absolute().joinpath(*PurePosixPath(name).parts)
    for path in (destination, *destination.parents):
        try:
            if _is_link(path):
                raise CheckpointError(f'Refusing a symlink/reparse-point output path: {path}')
        except FileNotFoundError:
            continue
    return destination


def _matches(path: Path, spec: FileSpec) -> bool:
    if not path.is_file():
        return False
    if spec.size_bytes is not None and path.stat().st_size != spec.size_bytes:
        return False
    with path.open('rb') as stream:
        digest, _ = _stream_digest(stream, limit=spec.size_bytes if spec.size_bytes is not None else MAX_EXTRACTED_BYTES)
    return digest == spec.sha256


def verify_files(manifest: Manifest, output: Path, *, require_all: bool = True) -> list[FileSpec]:
    """Return missing files; refuse existing files with different contents."""
    missing = []
    for spec in manifest.files:
        destination = _destination(output, spec.path)
        if not destination.exists():
            missing.append(spec)
        elif not _matches(destination, spec):
            raise CheckpointError(f'Existing checkpoint differs; refusing to overwrite: {destination}')
    if missing and require_all:
        raise CheckpointError(f'Missing {len(missing)} checkpoint file(s); first: {missing[0].path}. '
                              'Run this script without --verify-only to download them.')
    return missing


def _check_output_manifest(manifest: Manifest, output: Path) -> Path:
    destination = _destination(output, 'manifest.json')
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size != len(manifest.source_bytes):
            raise CheckpointError(f'Existing output manifest differs; refusing to overwrite: {destination}')
        with destination.open('rb') as stream:
            if stream.read(MAX_METADATA_BYTES + 1) != manifest.source_bytes:
                raise CheckpointError(f'Existing output manifest differs; refusing to overwrite: {destination}')
    return destination


def _install_output_manifest(manifest: Manifest, output: Path) -> None:
    destination = _check_output_manifest(manifest, output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = _check_output_manifest(manifest, output)
    try:
        stream = destination.open('xb')
    except FileExistsError:
        _check_output_manifest(manifest, output)
        return
    try:
        with stream:
            stream.write(manifest.source_bytes)
    except BaseException:
        destination.unlink()
        raise


def _credential_command(command: list[str], *, input_text: str | None = None) -> str | None:
    environment = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never',
                       GH_PROMPT_DISABLED='1', GIT_ASKPASS='', SSH_ASKPASS='')
    try:
        result = subprocess.run(command, input=input_text, text=True, capture_output=True,
                                timeout=15, check=False, env=environment)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def get_token(repository: str) -> str | None:
    """Read an existing credential without printing it or initiating login."""
    for variable in ('GITHUB_TOKEN', 'GH_TOKEN'):
        token = os.environ.get(variable, '').strip()
        if token and '\n' not in token and '\r' not in token:
            return token
    token = _credential_command(['gh', 'auth', 'token', '--hostname', 'github.com'])
    if token and '\n' not in token and '\r' not in token:
        return token
    credentials = _credential_command(
        ['git', '-c', 'core.askPass=', '-c', 'credential.interactive=false', 'credential', 'fill'],
        input_text=f'protocol=https\nhost=github.com\npath={repository}.git\n\n')
    if credentials:
        for line in credentials.splitlines():
            if line.startswith('password=') and line[9:]:
                return line[9:]
    return None


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """GitHub's signed asset redirect must not receive the GitHub credential."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        previous = urllib.parse.urlsplit(req.full_url)
        redirected = urllib.parse.urlsplit(newurl)
        if redirected.scheme != 'https' or not redirected.hostname or redirected.username or redirected.password:
            raise CheckpointError('Refusing an unsafe release-download redirect.')
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if result is not None and (previous.hostname, previous.port) != (redirected.hostname, redirected.port):
            result.remove_header('Authorization')
            result.remove_header('Cookie')
        return result


def _request(opener, url: str, token: str | None, accept: str):
    headers = {'Accept': accept, 'User-Agent': 'CORE-PIVOT-checkpoint-downloader',
               'X-GitHub-Api-Version': '2022-11-28'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    request = urllib.request.Request(url, headers=headers)
    try:
        return opener.open(request, timeout=TIMEOUT_SECONDS)
    except urllib.error.HTTPError as exc:
        raise CheckpointError(f'GitHub request failed (HTTP {exc.code}). For a private repository, '
                              'provide GITHUB_TOKEN/GH_TOKEN or an existing gh/Git login with read access.') from None
    except (OSError, urllib.error.URLError) as exc:
        raise CheckpointError('Unable to contact GitHub; check connectivity or use --archive for offline import.') from None


def download_archive(manifest: Manifest, destination: Path) -> None:
    token = get_token(manifest.repository)
    opener = urllib.request.build_opener(SafeRedirectHandler())
    base = f'https://api.github.com/repos/{manifest.repository}'
    tag = urllib.parse.quote(manifest.release_tag, safe='')
    with _request(opener, f'{base}/releases/tags/{tag}', token, 'application/vnd.github+json') as response:
        raw = response.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise CheckpointError('GitHub release metadata exceeds its size limit.')
    try:
        release = json.loads(raw)
    except (ValueError, UnicodeError):
        raise CheckpointError('GitHub returned invalid release metadata.') from None
    assets = release.get('assets') if isinstance(release, dict) else None
    if not isinstance(assets, list):
        raise CheckpointError('GitHub release metadata does not list assets.')
    matches = [asset for asset in assets if isinstance(asset, dict) and asset.get('name') == manifest.asset_name]
    if len(matches) != 1:
        raise CheckpointError(f'Expected exactly one release asset named {manifest.asset_name}.')
    asset = matches[0]
    if type(asset.get('id')) is not int or asset['id'] <= 0 or type(asset.get('size')) is not int or asset['size'] != manifest.asset_size_bytes:
        raise CheckpointError('GitHub release asset ID/size does not match the manifest.')
    with _request(opener, f"{base}/releases/assets/{asset['id']}", token, 'application/octet-stream') as response:
        length = response.headers.get('Content-Length')
        if length is not None and length != str(manifest.asset_size_bytes):
            raise CheckpointError('Release download Content-Length differs from the manifest.')
        with destination.open('xb') as stream:
            digest, size = _stream_digest(response, limit=manifest.asset_size_bytes, destination=stream)
    if size != manifest.asset_size_bytes or digest != manifest.asset_sha256:
        raise CheckpointError('Downloaded archive size/SHA-256 differs from the manifest.')


def install_archive(manifest: Manifest, archive: Path, output: Path) -> int:
    """Verify the entire archive and every member before installing any file."""
    missing = verify_files(manifest, output, require_all=False)
    with archive.open('rb') as stream:
        digest, size = _stream_digest(stream, limit=manifest.asset_size_bytes)
        if size != manifest.asset_size_bytes or digest != manifest.asset_sha256:
            raise CheckpointError('Archive size/SHA-256 differs from the manifest.')
        stream.seek(0)
        try:
            with zipfile.ZipFile(stream) as zipped:
                expected = {spec.path: spec for spec in manifest.files}
                members = zipped.infolist()
                names = set()
                total_size = 0
                for member in members:
                    if member.orig_filename != member.filename:
                        raise CheckpointError('ZIP member contains a truncated/unsafe filename.')
                    name = _safe_path(member.filename)
                    if name in names:
                        raise CheckpointError(f'Duplicate ZIP member: {name}')
                    names.add(name)
                    if name not in expected:
                        raise CheckpointError(f'Unexpected ZIP member: {name}')
                    kind = stat.S_IFMT(member.external_attr >> 16)
                    if member.is_dir() or kind not in (0, stat.S_IFREG) or member.flag_bits & 1:
                        raise CheckpointError(f'ZIP member is not an unencrypted regular file: {name}')
                    spec = expected[name]
                    if member.file_size < 0 or (spec.size_bytes is not None and member.file_size != spec.size_bytes):
                        raise CheckpointError(f'ZIP member size differs from the manifest: {name}')
                    total_size += member.file_size
                    if total_size > MAX_EXTRACTED_BYTES:
                        raise CheckpointError('ZIP exceeds the extracted size limit.')
                if names != set(expected):
                    raise CheckpointError('ZIP is missing one or more manifest checkpoint files.')
                with tempfile.TemporaryDirectory(prefix='core-pivot-checkpoints-') as staging_name:
                    staging = Path(staging_name)
                    for index, member in enumerate(members):
                        with zipped.open(member) as source, (staging / str(index)).open('xb') as target:
                            digest, size = _stream_digest(source, limit=member.file_size, destination=target)
                        if size != member.file_size or digest != expected[member.filename].sha256:
                            raise CheckpointError(f'ZIP checkpoint size/SHA-256 mismatch: {member.filename}')
                    # Recheck everything after staging, before the first installation.
                    missing = verify_files(manifest, output, require_all=False)
                    needed = {spec.path for spec in missing}
                    installed = 0
                    for index, member in enumerate(members):
                        if member.filename not in needed:
                            continue
                        destination = _destination(output, member.filename)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination = _destination(output, member.filename)
                        try:
                            target = destination.open('xb')
                        except FileExistsError:
                            if not _matches(destination, expected[member.filename]):
                                raise CheckpointError(f'Checkpoint appeared during import; refusing to overwrite: {destination}') from None
                            continue
                        try:
                            with target, (staging / str(index)).open('rb') as source:
                                shutil.copyfileobj(source, target, CHUNK_BYTES)
                        except BaseException:
                            destination.unlink()
                            raise
                        installed += 1
        except (zipfile.BadZipFile, NotImplementedError, RuntimeError) as exc:
            raise CheckpointError('Invalid, corrupted, encrypted, or unsupported checkpoint ZIP archive.') from exc
    verify_files(manifest, output)
    return installed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path,
                        help='Destination directory, including a manifest.json copy (default: manifest directory).')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--verify-only', action='store_true', help='Check local files only; never access the network.')
    modes.add_argument('--archive', type=Path, help='Import an already downloaded ZIP without network access.')
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        output = args.output if args.output is not None else args.manifest.resolve().parent
        if args.output is not None and not args.verify_only:
            _check_output_manifest(manifest, output)
        missing = verify_files(manifest, output, require_all=args.verify_only)
        installed = 0
        if missing:
            if args.archive is not None:
                installed = install_archive(manifest, args.archive, output)
            else:
                with tempfile.TemporaryDirectory(prefix='core-pivot-download-') as temporary:
                    archive = Path(temporary) / manifest.asset_name
                    print(f'Downloading {manifest.asset_name} ({manifest.asset_size_bytes:,} bytes).')
                    download_archive(manifest, archive)
                    installed = install_archive(manifest, archive, output)
        if args.output is not None and not args.verify_only:
            _install_output_manifest(manifest, output)
        print(f'Verified {len(manifest.files)} unique files for {manifest.checkpoint_count} selected checkpoints; '
              f'installed {installed}. Location: {output.absolute()}')
        return 0
    except CheckpointError as exc:
        print(f'Checkpoint error: {exc}', file=sys.stderr)
        return 1
    except (OSError, EOFError, ValueError):
        # Do not echo arbitrary transport exceptions, URLs, or credential-helper output.
        print('Checkpoint error: unable to read/write checkpoint files or complete the transfer.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
