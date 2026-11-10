"""Validate the sealed scientific design and unchanged numerical kernels; no GPU import."""
from pathlib import Path
import argparse,ast,hashlib,json
W=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def function(path,name):
 tree=ast.parse(path.read_text())
 return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
def files():
 top=[p for p in W.iterdir() if p.is_file() and p.suffix in ('.py','.json','.md','.sh','.ps1','.diff') and p.name not in ('code_manifest.json','work_state.json')]
 return sorted(top+[p for p in (W/'references').rglob('*') if p.is_file()])
def validate(seal=False):
 s=json.loads((W/'settings.json').read_text());old=json.loads((W/'references/exp033_settings.json').read_text())
 assert s['training_seeds']==[dict(id=f's{i}',init=i,train=i-3) for i in (6,7,8)]
 assert s['arms']==s['new_arms']==['A','P25','PR25','C25']
 assert s['relaxation']=={'P25':.25,'PR25':.25,'C25':.25}
 assert s['max_compute_seconds']==36000 and s['chat_monitoring'] is False and s['automatic_retry'] is False
 assert s['active_data_sets']==list(s['data_sets'])
 assert s['data_sets']['mechanism_discovery']=={'seed':20261102,'n':1000}
 assert s['discovery_generator_manifest_sha256']==sha(W/'discovery_creation_manifest.json')
 generator=json.loads((W/'discovery_creation_manifest.json').read_text())
 for name,digest in generator['generator_files'].items():assert sha(W/name)==digest
 assert set(s['preserved_data_sha256'])=={'protection.h5','monitor.h5','confirm.h5','mechanism_confirm.h5'}
 for k in ['start_stage','end_stage','train_bs','train_microbs','protection_bs_per_task','protection_tasks','fresh_tasks','eval_bs','mechanism_bs','n','confirm_stages','eval_every','monitor_every','resume_every','min_icl_gain','ni_acc_tolerance','ni_ce_tolerance','diagnostic_update_offsets','forward_seed','audit_rows']:
  assert s[k]==old[k],k
 for k in old:
  if k.startswith(('numeric_','mechanism_')):assert s[k]==old[k],k
 for name in ['modified_step','constrain_update','functional_objective','adam_update_from_state','eval_batch','mechanism_batch','mechanism_batch_inner']:
  assert function(W/'runtime.py',name)==function(W/'references/exp033_runtime.py',name),name
 for name in ['load_opts','template','load_checkpoint','initial_checkpoint','base_checkpoint','make_sampler']:
  assert function(W/'common.py',name)==function(W/'references/exp033_common.py',name),name
 for module in ['projection.py','mechanism.py']:
  assert sha(W/module)==sha(W/'references'/('exp033_'+module)),module
 for p in files():
  if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
  if p.suffix=='.json':json.loads(p.read_text())
 manifest={str(p.relative_to(W)):sha(p) for p in files()}
 if seal:(W/'code_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 else:assert manifest==json.loads((W/'code_manifest.json').read_text()),'sealed files changed'
 r=dict(passed=True,files=len(manifest),unchanged_numerical_kernels=7,
  unchanged_initialization_sampler_functions=6,code_manifest_sha256=sha(W/'code_manifest.json'),
  seed_ids=[x['id'] for x in s['training_seeds']],arms=s['arms'],scientific_compute=False)
 (W/'method_validation').mkdir(exist_ok=True)
 (W/'method_validation/package_check.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r));return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');a=p.parse_args();validate(a.seal)
