"""Explicit asset export; no model execution or dependency installation."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
from importlib.metadata import version as distribution_version


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def export(root, destination):
    work = Path(__file__).resolve().parent
    spec = json.loads((work / 'input_spec.json').read_text())
    if destination.exists():
        raise RuntimeError('destination exists; refusing to overwrite assets')
    destination.mkdir(parents=True)
    records = {}
    for name, item in spec['files'].items():
        source, target = root / item['from'], destination / name
        digest = sha(source)
        if digest != item['sha256']:
            raise ValueError('source asset identity failed: ' + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(target) != digest:
            raise ValueError('copy mismatch: ' + name)
        records[name] = dict(sha256=digest, size=target.stat().st_size)
    item = spec['replay_checkpoint']
    target = destination / item['to']
    shutil.copyfile(root / item['from'], target)
    records[item['to']] = dict(sha256=sha(target), size=target.stat().st_size,
                              newly_registered=True)
    if item.get('sha256') is not None and records[item['to']]['sha256']!=item['sha256']:
        raise ValueError('registered replay checkpoint changed')
    # Author source imports its utils module; copy other *.py under same frozen
    # source directory and register their actual hashes. No unrelated folders.
    for source in sorted((root / 'reproduction_support/source').glob('*.py')):
        name = 'source/' + source.name
        if name in records:
            continue
        target = destination / name
        shutil.copyfile(source, target)
        records[name] = dict(sha256=sha(target), size=target.stat().st_size,
                              newly_registered=True)
    (destination / 'assets_manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    if spec.get('registered_asset_manifest_sha256') is not None:
        if sha(destination/'assets_manifest.json')!=spec['registered_asset_manifest_sha256']:
            raise ValueError('exported complete manifest differs from registered assets')
    versions={name:distribution_version(name) for name in ['jax','jaxlib','equinox','optax','numpy','h5py']}
    print(json.dumps(dict(files=len(records), bytes=sum(v['size'] for v in records.values()),
                         manifest_sha256=sha(destination / 'assets_manifest.json'),versions=versions)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    export(args.project_root.resolve(), args.destination.resolve())
