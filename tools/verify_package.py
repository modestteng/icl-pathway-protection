"""Check an exact manuscript-scoped release without importing scientific code."""
from pathlib import Path
import argparse
import ast
import hashlib
import io
import json
import re
import struct
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = {
    'private_filesystem_prefix': re.compile(r'/(?:'+'|'.join(('Users','home','root'))+r')/|/mnt/[a-z]/|(?<![A-Za-z0-9])[A-Za-z]:[\\/]'),
    'email_address': re.compile(r'[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}'),
    'credential': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}'),
    'private_network': re.compile(r'\b(?:100\.\d+\.\d+\.\d+|127\.0\.0\.1|192\.168\.\d+\.\d+)\b'),
}
FORBIDDEN_PARTS = {'.git','.DS_Store','__pycache__','.env','node_modules','.venv','venv','history','backup','backups'}
FORBIDDEN_SUFFIXES = {'.zip','.gz','.tar','.pdf','.doc','.docx','.pptx','.jpg','.jpeg','.png','.heic','.key','.pem','.log','.jsonl','.ps1'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-upstream',action='store_true',
                        help='Also require locally restored, hash-matched upstream source.')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    inventory=json.loads((ROOT/'reproducibility/paper_inventory.json').read_text())
    dependency=json.loads((ROOT/'reproducibility/upstream_dependency.json').read_text())
    external={dest:rec['sha256'] for rec in dependency['files'] for dest in rec['destinations']}
    allowed_families=inventory['experimental_families']
    errors=[];syntax_count=0;npz_count=0;arrays=0;scanned=0
    def error(path,problem):
        errors.append({'path':str(path),'problem':problem})
    def scan(path,text):
        nonlocal scanned
        scanned+=1
        for kind,pattern in PRIVATE.items():
            if pattern.search(text):error(path,kind)
    expected={r['path']:r for r in manifest['files']}
    if len(expected)!=len(manifest['files']):error('MANIFEST.json','duplicate manifest path')
    if set(expected)&set(external):error('MANIFEST.json','upstream source must remain external')
    # A Git clone includes repository metadata outside the released file set.
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
            if p.is_file() and p.relative_to(ROOT).parts[0] != '.git'}
    for rel in sorted(actual-expected.keys()-external.keys()-{'MANIFEST.json'}):error(rel,'not on release allowlist')
    for rel in sorted(expected.keys()-actual):error(rel,'missing allowed file')
    upstream_present=0
    for rel,digest in external.items():
        p=ROOT/rel
        if p.exists() or p.is_symlink():
            if p.is_symlink() or not p.is_file():error(rel,'invalid external source file')
            elif hashlib.sha256(p.read_bytes()).hexdigest()!=digest:error(rel,'external source SHA-256 mismatch')
            else:upstream_present+=1
        elif args.require_upstream:error(rel,'required external source missing; see UPSTREAM_SETUP.md')
    for rel,record in expected.items():
        parts=Path(rel).parts;p=ROOT/rel
        if not parts or '..' in parts or Path(rel).is_absolute():
            error(rel,'unsafe manifest path');continue
        if any(x in FORBIDDEN_PARTS for x in parts) or p.suffix.lower() in FORBIDDEN_SUFFIXES:
            error(rel,'excluded personal, historical, or operational file type')
        if p.is_symlink():error(rel,'symlink is not a release asset')
        if parts[0]=='experiments' and parts[1]!='reproduction_support':
            family=allowed_families.get(parts[1])
            if family is None:error(rel,'experiment absent from manuscript scope')
            elif record.get('scope')!=family['id']:error(rel,'incorrect experiment scope binding')
        if not p.is_file():continue
        data=p.read_bytes()
        if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:
            error(rel,'size or SHA-256 mismatch')
        if p.suffix=='.py':
            try:ast.parse(data.decode(),filename=rel);syntax_count+=1
            except (SyntaxError,UnicodeError):error(rel,'invalid Python syntax')
        if p.suffix=='.npz':
            npz_count+=1
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    if z.testzip():error(rel,'NPZ CRC mismatch')
                    for member in z.namelist():
                        arrays+=1;scan(rel+'::'+member,member)
                        b=z.read(member)
                        if not b.startswith(b'\x93NUMPY'):error(rel,'unexpected non-NPY array member');continue
                        version=b[6]
                        n=int.from_bytes(b[8:10] if version==1 else b[8:12],'little')
                        start=10 if version==1 else 12
                        header=ast.literal_eval(b[start:start+n].decode('latin1').strip())
                        dtype=header['descr'];scan(rel+'::header',str(header))
                        if isinstance(dtype,str) and 'O' in dtype:error(rel,'object arrays are excluded')
                        if isinstance(dtype,str) and dtype.lstrip('<>=|').startswith(('S','U')):
                            body=b[start+n:]
                            if 'U' in dtype:
                                text=body.decode('utf-32-be' if dtype.startswith('>') else 'utf-32-le')
                            else:text=body.decode('utf-8',errors='replace')
                            scan(rel+'::string_array',text)
            except Exception as exc:error(rel,'invalid NPZ: '+type(exc).__name__)
        else:
            try:scan(rel,data.decode('utf-8'))
            except UnicodeError:error(rel,'unreviewed binary file')
    scan('MANIFEST.json',(ROOT/'MANIFEST.json').read_text())
    for family in allowed_families:
        cfg=ROOT/'experiments'/family/'settings.json'
        if cfg.exists():
            d=json.loads(cfg.read_text())
            if 'fresh_tasks' in d and not set(d['fresh_tasks'])<=set(('icl','iwl_copy_avail','train_eval')):
                error(cfg.relative_to(ROOT),'evaluation preset includes an unreported task')
    if sorted(p.name for p in (ROOT/'paper_figures/svg').glob('*.svg'))!=[f'Figure_{i}.svg' for i in range(1,9)]:
        error('paper_figures/svg','figures differ from the eight current manuscript figures')
    d=json.loads((ROOT/'experiments/exp_036_multiseed_path_protection/settings.json').read_text())
    if [x['id'] for x in d['training_seeds']]!=['s7']:error('s7 replication','unreported training seed')
    report={'passed':not errors,'files_verified':len(expected),'python_files_parsed':syntax_count,
            'npz_archives_checked':npz_count,'array_members_checked':arrays,'privacy_text_units_scanned':scanned,
            'experimental_scope_entries':len(allowed_families),'scientific_code_imported':False,
            'upstream_source_destinations_verified':upstream_present,
            'upstream_source_destinations_expected':len(external),
            'upstream_source_required':args.require_upstream,
            'model_or_training_executed':False,'errors':errors}
    print(json.dumps(report,indent=2))
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
