"""Small CPU-only helpers. Importing this file never initializes CUDA."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'settings.json').read_text())
OUT = ROOT / 'outputs'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def verify_package():
    manifest = json.loads((ROOT / 'code_manifest.json').read_text())
    for name, expected in manifest.items():
        if sha(ROOT / name) != expected:
            raise RuntimeError('sealed file changed: ' + name)
    return sha(ROOT / 'code_manifest.json')
