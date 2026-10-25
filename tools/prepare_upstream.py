"""Restore hash-pinned upstream dependencies locally without executing them."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import tempfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--source-dir',type=Path,help='An existing checkout of the original upstream repository.')
    mode.add_argument('--fetch',action='store_true',help='Download only the seven pinned files from the original repository.')
    args=parser.parse_args()
    spec=json.loads((ROOT/'reproducibility/upstream_dependency.json').read_text())
    prepared=[]
    # Validate every dependency and destination before writing any file.
    for rec in spec['files']:
        if args.source_dir:
            data=(args.source_dir/rec['upstream_path']).read_bytes()
        else:
            url=spec['raw_base_url']+'/'+rec['upstream_path']
            with urllib.request.urlopen(url,timeout=30) as response:
                data=response.read()
        if hashlib.sha256(data).hexdigest()!=rec['sha256']:
            raise ValueError('Upstream SHA-256 mismatch: '+rec['upstream_path'])
        for rel in rec['destinations']:
            relative=Path(rel)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe dependency destination')
            target=ROOT/relative
            if target.is_symlink() or any(p.is_symlink() for p in target.parents if p!=ROOT.parent):
                raise ValueError('Symlink dependency destination: '+rel)
            if target.exists() and target.read_bytes()!=data:
                raise FileExistsError('Refusing to replace a different local file: '+rel)
            prepared.append((target,data))
    written=0
    for target,data in prepared:
        if target.exists():continue
        target.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as handle:
            temporary=Path(handle.name)
            handle.write(data)
        os.replace(temporary,target)
        written+=1
    print(json.dumps({'passed':True,'upstream_commit':spec['commit'],
                     'source_files_verified':len(spec['files']),
                     'destinations_verified':len(prepared),'destinations_written':written,
                     'source_executed':False,'dependencies_installed':False},indent=2))

if __name__=='__main__':main()
