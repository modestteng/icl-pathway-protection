"""Static sealing and P25 scientific-design inheritance checks; no JAX import."""
from pathlib import Path
import ast
import hashlib
import json
import argparse

W=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def files():
    top=[p for p in W.iterdir() if p.suffix in ['.py','.json','.md','.ps1','.sh','.diff'] and p.name not in ['code_manifest.json','work_state.json']]
    return sorted(top+[p for p in (W/'references').rglob('*') if p.is_file()])
def validate(seal=False):
    s=json.loads((W/'settings.json').read_text());p=json.loads((W/'references/exp031/settings.json').read_text())
    exceptions={'version','arms','relaxation','protection_mode','reference_arms','windows_estimate_hours','protected_path_head','reference_p25_work','reference_p25_receipt_sha256','performance_mode','gpu_memory_fraction'}
    for key in set(p)-exceptions:assert s[key]==p[key],key
    assert s['arms']==['PR25'] and s['relaxation']=={'PR25':.25}
    assert s['candidate_path_head']==0 and s['protected_path_head']==1
    assert s['ordinary_task_loss_constraints'] is False and s['max_compute_seconds']==10500
    inherited=json.loads((W/'source_inheritance.json').read_text())
    for name,item in inherited.items():assert item['unchanged'] and sha(W/name)==item['parent_sha256']
    for name,h in s['reference_receipt_sha256'].items():assert sha(W/'references'/name)==h
    for name,h in s['reference_p25_receipt_sha256'].items():assert sha(W/'references/exp031'/name)==h
    original=ast.parse((W/'references/exp031/runtime.py').read_text())
    current=ast.parse((W/'runtime.py').read_text())
    for name in ['modified_step','constrain_update','functional_objective','adam_update_from_state','eval_batch','mechanism_batch','mechanism_batch_inner']:
        a=next(n for n in original.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
        b=next(n for n in current.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),name
    for path in files():
        if path.suffix=='.py':ast.parse(path.read_text(),filename=str(path))
        if path.suffix=='.json':json.loads(path.read_text())
    manifest={str(path.relative_to(W)):sha(path) for path in files()}
    if seal:(W/'code_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    else:assert json.loads((W/'code_manifest.json').read_text())==manifest
    result=dict(passed=True,files=len(manifest),unchanged_numeric_kernels=7,protected_head=1,code_manifest_sha256=sha(W/'code_manifest.json'),scientific_compute=False)
    (W/'method_validation').mkdir(exist_ok=True);(W/'method_validation/package_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');a=p.parse_args();validate(a.seal)
