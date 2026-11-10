"""Independent saved-array verification; no GPU, no training, no imports of driver."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as handle:
  for block in iter(lambda:handle.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def choose(coverages):
 finite=[(i+1,c) for i,c in enumerate(coverages) if c is not None and math.isfinite(c)]
 if not finite:return 1,'no_finite_coverage_default_top1'
 reached=[k for k,c in finite if c>=.5]
 if reached:return min(reached),'smallest_prefix_reaching_target'
 top=max(c for k,c in finite)
 return min(k for k,c in finite if c==top),'target_not_reached_max_coverage_prefix'
def coverage(row):
 ratios=[]
 for task in ['icl','flip_icl']:
  native=np.asarray(row[task+'/native/donor_ce'],np.float64)
  donor=np.asarray(row[task+'/donor/donor_ce'],np.float64)
  path=np.asarray(row[task+'/path/donor_ce'],np.float64)
  denom=float((native-donor).mean())
  if denom<=0:return None
  ratios.append(float((native-path).mean())/denom)
 return min(ratios)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
 root=args.root.resolve();source=Path(__file__).resolve().parent
 protocol=read(source/'protocol.json');identity=sha(source/'code_manifest.json');out=root/'results';summary=read(out/'summary.json')
 execution=root/'execution_identity.json'
 if execution.exists():identity=read(execution)['code_manifest_sha256']
 expected=list(protocol['seeds']);assert set(expected)==set(summary['seeds'])
 N=(protocol['architecture']['depth']-1)*protocol['architecture']['num_heads'];result={'passed':True,'code_manifest_sha256':identity,'independent_saved_array_readback':True,'inference_rerun':False,'seeds':{}}
 assert summary['code_manifest_sha256']==identity and read(out/'depth/validation.json')['passed']
 all_diffs={}
 for sid in expected:
  d=out/'depth'/sid;timing=read(d/'timing.json');curve=read(d/'normal/curve.json')['points'];assert len(curve)==201 and [r['stage'] for r in curve]==list(range(0,20000001,100000))
  best=min(curve,key=lambda r:(-r['correct_count'],r['stage']));assert best['stage']==timing['start_stage'] and timing['end_stage']==best['stage']+5600000
  fork=d/'normal/checkpoints'/f"{timing['start_stage']:011d}.eqx";assert sha(fork)==timing['start_checkpoint_sha256']
  selected=read(d/'discovery/selection.json');assert selected['seed']==protocol['seeds'][sid];assert selected['candidate_count']==N and selected['candidates']==list(range(N)) and selected['artificial_cap'] is None
  assert not selected['mechanism_confirm_used_for_selection'] and not selected['confirm_used_for_selection']
  assert selected['fork_checkpoint_sha256']==timing['start_checkpoint_sha256'] and selected['rho']==.25
  scores={}
  for c in selected['candidates']:
   with np.load(d/'discovery/single'/f'{c}.npz',allow_pickle=False) as r:
    scores[c]=float(np.mean([(np.asarray(r[t+'/native/donor_ce'],np.float64)-np.asarray(r[t+'/path/donor_ce'],np.float64)).mean() for t in ['icl','flip_icl']]))
  ranking=sorted(scores,key=lambda c:(-scores[c],c));assert ranking==selected['ranking']
  cov=[]
  for k in range(1,N+1):
   with np.load(d/'discovery'/f'prefix{k}.npz',allow_pickle=False) as r:
    assert list(r['path_set'])==sorted(ranking[:k]);cov.append(coverage(r))
  assert all((a is None and b is None) or a is not None and b is not None and math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-8) for a,b in zip(cov,selected['prefix_coverage']))
  k,reason=choose(cov);assert k==selected['k'] and reason==selected['selection_reason'] and selected['selected']==sorted(ranking[:k])
  q=read(d/'discovery/queue_validation.json');assert q['passed'] and q['code_manifest_sha256']==identity
  stages=list(range(timing['end_stage']-1000000,timing['end_stage']+1,100000));means={};identities={}
  for arm in ['A','P25_uncapped']:
   assert read(d/arm/'complete.json')['passed'];collect={}
   for stage in stages:
    path=d/arm/'confirm'/f'{arm}_{stage:011d}.npz'
    with np.load(path,allow_pickle=False) as r:
     assert str(r['code_manifest_sha256'])==identity and int(r['stage'])==stage and str(r['seed_id'])==sid
     for task in ['train_eval','iwl_copy_avail','icl']:
      key_ids=task+'/identities'
      if key_ids in r:
       if task in identities:assert np.array_equal(identities[task],r[key_ids])
       else:identities[task]=r[key_ids].copy()
      for metric in ['acc','loss']:
       key=task+'/'+metric;a=np.asarray(r[key],np.float64);assert a.shape==(5000,) and np.isfinite(a).all()
       if metric=='acc':assert np.isin(a,[0,1]).all()
       collect.setdefault(key,[]).append(float(a.mean()))
    assert sha(path)==summary['seeds'][sid][arm]['confirm_files_sha256'][str(path.relative_to(out))]
   means[arm]={key:float(np.mean(v)) for key,v in collect.items()}
   assert all(math.isclose(value,summary['seeds'][sid][arm]['means'][key],abs_tol=1e-12) for key,value in means[arm].items())
  diffs={key:means['P25_uncapped'][key]-means['A'][key] for key in means['A']}
  for key,value in diffs.items():
   assert math.isclose(value,summary['seeds'][sid]['paired_differences'][key],abs_tol=1e-12);all_diffs.setdefault(key,[]).append(value)
  result['seeds'][sid]={'start_stage':timing['start_stage'],'end_stage':timing['end_stage'],'candidate_count':N,'selected_count':k,'selection_reason':reason,'means':means,'paired_differences':diffs}
 result['across_seed_mean_differences']={key:float(np.mean(v)) for key,v in all_diffs.items()}
 for key,value in result['across_seed_mean_differences'].items():assert math.isclose(value,summary['across_seed_mean_differences'][key],abs_tol=1e-12)
 (root/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
