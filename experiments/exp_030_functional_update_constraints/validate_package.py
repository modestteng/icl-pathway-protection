"""Standard-library-only artifact check; never imports scientific runtime."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import tarfile

WORK = Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()


def code_files():
    return sorted([p for p in WORK.iterdir() if p.suffix=='.py']
        + [WORK/n for n in ['settings.json','input_spec.json','data_creation_manifest.json','protocol.md',
                            'cloud_handoff.md','launch_cloud.sh','status_cloud.sh','export_assets.ps1',
                            'stage_asset_archive.ps1','run_windows.sh','run_windows.ps1',
                            'register_windows.ps1','status_windows.ps1','deploy_windows.ps1']])


def validate(code_only=False, seal=False):
    files=code_files()
    for p in files:
        if p.suffix=='.py':
            ast.parse(p.read_text(encoding='utf-8'),filename=p.name)
        elif p.suffix=='.json':
            json.loads(p.read_text(encoding='utf-8'))
    settings=json.loads((WORK/'settings.json').read_text())
    assert settings['analysis_scope']=='single_seed_pilot'
    assert settings['arms']==['A','B','C25','D25','R25']
    assert settings['training_seeds']==[{'id':'s5','init':5,'train':2}]
    assert len(settings['confirm_stages'])==11
    assert settings['late_window']==[9000000,10000000]
    assert settings['start_stage']==4400000 and settings['end_stage']==10000000
    assert settings['train_bs']==settings['train_microbs']==32
    assert settings['relaxation']=={'C25':.25,'D25':.25,'R25':.25}
    assert settings['compute_route']=='windows_user_requested'
    assert not settings['require_seed_aggregate_holm']
    assert settings['active_data_sets']==['mechanism_confirm','protection','monitor','confirm']
    assert sha(WORK/'data_creation_manifest.json')==settings['data_generator_manifest_sha256']
    assert set(settings['preserved_data_sha256'])=={n+'.h5' for n in settings['data_sets']}
    manifest={p.name:sha(p) for p in files}
    if seal:
        (WORK/'code_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    else:
        assert json.loads((WORK/'code_manifest.json').read_text())==manifest,'sealed source changed'
    result=dict(code_files=len(files),code_manifest_sha256=sha(WORK/'code_manifest.json'),
                syntax_and_configuration=True,scientific_compute=False,
                gpu_preflight='NOT_RUN',functional_metric_validation='NOT_RUN')
    if not code_only:
        assets=Path(os.environ.get('ICL030_ASSETS',str(WORK/'assets')))
        spec=json.loads((WORK/'input_spec.json').read_text())
        for name,item in spec['files'].items():
            assert sha(assets/name)==item['sha256'],name
        exported=json.loads((assets/'assets_manifest.json').read_text())
        assert sha(assets/'assets_manifest.json')==spec['registered_asset_manifest_sha256']
        assert spec['replay_checkpoint']['to'] in exported
        assert sha(assets/spec['replay_checkpoint']['to'])==spec['replay_checkpoint']['sha256']
        for name,item in exported.items():
            assert sha(assets/name)==item['sha256'],name
        result['assets_manifest_sha256']=sha(assets/'assets_manifest.json')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--code-only',action='store_true')
    parser.add_argument('--seal',action='store_true')
    parser.add_argument('--bundle',action='store_true')
    args=parser.parse_args()
    result=validate(args.code_only,args.seal)
    if args.bundle:
        destination=WORK/'exp030_source.tar.gz'
        with tarfile.open(destination,'w:gz') as archive:
            for p in code_files()+[WORK/'code_manifest.json']:
                archive.add(p,arcname='exp030/'+p.name)
            for name in ['projection_engineering_receipt.json','mechanism_review.json']:
                if (WORK/name).exists():
                    archive.add(WORK/name,arcname='exp030/'+name)
            for name in ['asset_registration.json','asset_archive_receipt.json']:
                if (WORK/name).exists():
                    archive.add(WORK/name,arcname='exp030/'+name)
            for path in sorted((WORK/'method_validation').glob('*.json')):
                archive.add(path,arcname='exp030/method_validation/'+path.name)
        result['bundle_sha256']=sha(destination)
        result['bundle_bytes']=destination.stat().st_size
    receipt=WORK/'method_validation/package_check.json'
    receipt.parent.mkdir(exist_ok=True)
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
