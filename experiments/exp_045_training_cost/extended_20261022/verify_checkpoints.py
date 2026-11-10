"""Independent CPU artifact readback after all timed GPU processes exit."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'settings.json').read_text())
os.environ['JAX_PLATFORMS']='cpu'
os.environ['ICL031_ASSETS']=str(Path(os.environ.get('ICL_EXPERIMENT_ROOT', str(ROOT.parent)))/'exp_030_functional_update_constraints/assets')
sys.path.insert(0,str(Path(os.environ.get('ICL_EXPERIMENT_ROOT', str(ROOT.parent)))/'exp_033_random_path_only_protection'))
import common as r
import numpy as np


def main():
    complete=json.loads((ROOT/'outputs/supervisor_complete.json').read_text())
    assert complete['passed']
    assert r.jax.default_backend()=='cpu'
    records=[]
    for result_file in sorted((ROOT/'outputs').glob('r*/result.json')):
        result=json.loads(result_file.read_text())
        p=result_file.parent/'final_state.eqx'
        assert r.sha(p)==result['final_checkpoint_sha256']
        ck=r.load_checkpoint(p,dict(id='s5',init=5,train=2))
        count=int(np.asarray(ck['opt_state'][0].count))
        assert int(ck['iter'])==result['end_stage']
        assert count==int(ck['iter'])//32
        keys={k:np.asarray(v).tolist() for k,v in ck['seeds'].items()}
        assert keys==result['final_prng']
        values=[np.asarray(v) for v in r.arrays(ck)]
        assert all(np.isfinite(v).all() for v in values)
        records.append(dict(tag=result['tag'],sha256=r.sha(p),iter=int(ck['iter']),
            adam_count=count,prng=keys,array_count=len(values),
            array_bytes=sum(v.nbytes for v in values),all_finite=True))
    assert len(records)==complete['rounds_done']*4
    # This is a readback check, not a rerun of the training computation.
    r.write_json(ROOT/'outputs/independent_checkpoint_readback.json',
        dict(passed=True,device='cpu',records=records,independent_training_rerun=False))
    print(json.dumps(dict(passed=True,checkpoints=len(records),device='cpu')))


if __name__=='__main__':
    main()
