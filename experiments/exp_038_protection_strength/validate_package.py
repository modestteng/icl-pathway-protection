"""Seal the runnable plan and minimal scientific adaptations without imports."""
from pathlib import Path
import argparse,ast,hashlib,json,tarfile

WORK=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected():
    return sorted(p for p in WORK.rglob('*') if p.is_file()
        and p.suffix in {'.py','.json','.md','.ps1','.sh'}
        and not any(part in {'outputs','logs','attempts','startup_checks','__pycache__'} for part in p.relative_to(WORK).parts)
        and p != WORK/'code_manifest.json' and p.name != 'package_check.json' )


def functions(path):
    text=Path(path).read_text();tree=ast.parse(text)
    return {n.name:ast.get_source_segment(text,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}


def validate(seal=False,bundle=False):
    settings=json.loads((WORK/'settings.json').read_text())
    assert settings['arms']==[k+s for s in ['10','50','75'] for k in ['P','C','D']]
    assert settings['relaxation']=={a:int(a[1:])/100 for a in settings['arms']}
    assert settings['max_compute_seconds']==100800 and not settings['automatic_retry'] and not settings['continuous_chat_supervision']
    assert settings['training_seeds']==[dict(id='s5',init=5,train=2)]
    assert settings['strength_grid']==[.1,.25,.5,.75]
    assert settings['start_stage']==4400000 and settings['end_stage']==10000000
    assert settings['train_bs']==settings['train_microbs']==settings['protection_bs_per_task']==32
    inherited=json.loads((WORK/'source_inheritance.json').read_text())
    for name in ['mechanism.py','projection.py','datasets.py','input_spec.json','data_creation_manifest.json','reference_analysis.py','reference_reader.py']:
        assert inherited[name]['unchanged'] and sha(WORK/name)==inherited[name]['parent_sha256'],name
    for name,digest in json.loads((WORK/'references_manifest.json').read_text()).items():
        assert sha(WORK/'references'/name)==digest,name
    for path in selected():
        if path.suffix=='.py':ast.parse(path.read_text(),filename=str(path))
        if path.suffix=='.json':json.loads(path.read_text())
    scientific=functions(WORK/'science.py')
    for name,digest in inherited['scientific_function_sha256'].items():
        assert hashlib.sha256(scientific[name].encode()).hexdigest()==digest,name
    for name,digest in inherited['dispatch_function_sha256'].items():
        assert hashlib.sha256(functions(WORK/'performance/engine.py')[name].encode()).hexdigest()==digest,name
    manifest={str(p.relative_to(WORK)):sha(p) for p in selected()}
    if seal:(WORK/'code_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    else:assert json.loads((WORK/'code_manifest.json').read_text())==manifest,'sealed source changed'
    result=dict(passed=True,files=len(manifest),code_manifest_sha256=sha(WORK/'code_manifest.json'),scientific_compute=False)
    if bundle:
        path=WORK/'method_validation/source_bundle.tar.gz'
        with tarfile.open(path,'w:gz') as archive:
            for name in [*manifest,'code_manifest.json']:
                archive.add(WORK/name,arcname=WORK.name+'/'+name,recursive=False)
        result.update(bundle_sha256=sha(path),bundle_bytes=path.stat().st_size)
    (WORK/'method_validation/package_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');parser.add_argument('--bundle',action='store_true')
    args=parser.parse_args();validate(args.seal,args.bundle)
