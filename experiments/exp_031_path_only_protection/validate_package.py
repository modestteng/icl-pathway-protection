"""Standard-library source/configuration sealing, no scientific computation."""
from pathlib import Path
import ast,json,hashlib,argparse,tarfile
W=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def files():
    top=[p for p in W.iterdir() if p.suffix in ['.py','.json','.md','.ps1','.sh','.diff'] and p.name not in ['code_manifest.json','work_state.json']]
    return sorted(top+list((W/'references').glob('*.json'))+list((W/'attempts/attempt_001_unused_discovery_overlap/source').iterdir()))
def validate(seal=False,bundle=False):
    s=json.loads((W/'settings.json').read_text())
    assert s['arms']==['P25'] and s['relaxation']=={'P25':.25}
    assert s['ordinary_task_loss_constraints'] is False and s['protection_mode']=='path_only'
    assert s['candidate_path_head']==0 and s['training_seeds']==[{'id':'s5','init':5,'train':2}]
    assert s['start_stage']==4400000 and s['end_stage']==10000000
    assert s['confirm_stages']==list(range(9000000,10000001,100000)) and s['late_window']==[9000000,10000000]
    assert s['train_bs']==s['train_microbs']==s['protection_bs_per_task']==32
    assert s['max_compute_seconds']==10500 and s['primary_gain_family_size']==2
    assert s['compute_route']=='windows_default_under_3h'
    inherit=json.loads((W/'source_inheritance.json').read_text())
    for n in ['mechanism.py','projection.py','datasets.py','input_spec.json','data_creation_manifest.json']:
        assert inherit[n]['unchanged'] and sha(W/n)==inherit[n]['parent_sha256']
    for n,h in s['reference_receipt_sha256'].items():assert sha(W/'references'/n)==h
    for p in files():
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
        if p.suffix=='.json':json.loads(p.read_text())
    runtime=ast.parse((W/'runtime.py').read_text());f=next(x for x in runtime.body if isinstance(x,ast.FunctionDef) and x.name=='modified_step')
    assert 'assert len(gradients) == len(losses) == 1' in ast.get_source_segment((W/'runtime.py').read_text(),f)
    m={str(p.relative_to(W)):sha(p) for p in files()}
    if seal:(W/'code_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    else:assert json.loads((W/'code_manifest.json').read_text())==m,'Sealed source changed'
    result={'passed':True,'files':len(m),'code_manifest_sha256':sha(W/'code_manifest.json'),'scientific_compute':False}
    if bundle:
        with tarfile.open(W/'exp031_source.tar.gz','w:gz') as z:
            for p in files()+[W/'code_manifest.json']:z.add(p,arcname='exp031/'+str(p.relative_to(W)))
        result['bundle_sha256']=sha(W/'exp031_source.tar.gz')
    (W/'method_validation').mkdir(exist_ok=True);(W/'method_validation/package_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');p.add_argument('--bundle',action='store_true');a=p.parse_args();validate(a.seal,a.bundle)
