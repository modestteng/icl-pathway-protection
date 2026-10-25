"""Independent saved-artifact reader; imports no producer, statistic or GPU module."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent
OUT = HERE/'outputs'
CHECKS, FAILURES = 0, []


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def check(value, name):
    global CHECKS
    CHECKS += 1
    if not bool(value):
        FAILURES.append(name)


def quantile(p):
    # Closed Student-t CDF for df=4, independent of the producer's beta routine.
    low, high = -10000., 10000.
    for _ in range(100):
        mid = (low+high)/2
        z = mid/math.sqrt(4+mid*mid)
        cdf = .5+(3*z-z**3)/4
        if cdf < p:
            low = mid
        else:
            high = mid
    return (low+high)/2


def verify_summary(values, saved, name):
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    se = sd/math.sqrt(5)
    check(saved['n'] == saved['expected_n'] == 5 and saved['complete'], name+'/complete')
    expected = dict(mean=mean, sd=sd, se=se, lower95=mean-quantile(.975)*se,
        upper95=mean+quantile(.975)*se, lower_one_sided95=mean-quantile(.95)*se,
        upper_one_sided95=mean+quantile(.95)*se)
    check(all(abs(a-b)<2e-10 for a,b in zip(values,saved['values'])), name+'/values')
    for key, value in expected.items():
        check(abs(saved[key]-value)<2e-10, name+'/'+key)


def verify():
    import numpy as np
    import h5py
    config = read(HERE/'supplement.json')
    settings = read(ORIGINAL/'settings.json')
    identity = sha(ORIGINAL/'code_manifest.json')
    supplement_identity = sha(HERE/'manifest.json')
    inventory = read(HERE/'original_inventory.json')
    for name, expected in read(HERE/'manifest.json').items():
        check(sha(HERE/name)==expected, 'supplement_source/'+name)
    for name, expected in read(ORIGINAL/'code_manifest.json').items():
        check(sha(ORIGINAL/name)==expected, 'original_source/'+name)
    check(identity==config['original_manifest_sha256'], 'original_identity')
    for name,key in [('analysis.json','original_analysis_sha256'),
                     ('verification.json','original_verification_sha256'),
                     ('runtime_complete.json','original_runtime_complete_sha256')]:
        check(sha(ORIGINAL/'outputs'/name)==inventory[key], 'original_immutable/'+name)
    check(read(ORIGINAL/'outputs/verification.json')['passed'], 'original_verification')
    for role,origin in inventory['origins'].items():
        check(sha(ORIGINAL/'outputs'/origin['checkpoint_path'])==origin['checkpoint_sha256'],
              'original_checkpoint_immutable/'+role)
    spec = read(ORIGINAL/'input_spec.json')
    for name, record in spec['files'].items():
        check(sha(ORIGINAL/'assets'/name)==record['sha256'], 'asset/'+name)
    for name in settings['data_sets']:
        expected = spec['files']['frozen_data/'+name+'.h5']['sha256']
        check(sha(OUT/'data'/(name+'.h5'))==expected, 'frozen_pool/'+name)
    tasks = settings['reported_tasks']
    pools = {}
    with h5py.File(OUT/'data/confirm.h5','r') as f:
        for task in tasks:
            pools[task] = (np.asarray(f[task]['identities']), np.asarray(f[task]['labels']))
    saved_rows = {}
    for sid in ['s21','s22']:
        for arm in ['A','P25']:
            for suffix in ['', '_diagnostic']:
                name = f'{sid}_{arm}_40M'+suffix
                job = OUT/'jobs'/name
                done, exit_record = read(job/'complete.json'), read(job/'exit.json')
                check(done['passed'] and done['code_manifest_sha256']==identity and
                    done['supplement_manifest_sha256']==supplement_identity, name+'/complete')
                check(exit_record['exit_code']==0 and exit_record['real_exit_confirmed'], name+'/real_exit')
            origin = read(OUT/sid/arm/'long_origin.json')
            original_origin = inventory['origins'][sid+'_'+arm]
            check(origin['state']==original_origin and origin['source_10M_sha256']==original_origin['checkpoint_sha256'],
                  sid+'/'+arm+'/origin_exact')
            acceptance = read(OUT/sid/'resume_acceptance'/(arm+'.json'))
            check(acceptance['passed'] and all(acceptance['checks'].values()), sid+'/'+arm+'/resume_acceptance')
            complete = read(OUT/sid/arm/'complete_40000000.json')
            check(complete['iter']==complete['evaluated_stage']==40000000, sid+'/'+arm+'/end')
            check(10000000<=complete['launch_start']<=40000000, sid+'/'+arm+'/continued_not_retrained')
            diagnostic = read(OUT/sid/arm/'mechanism_late/00040000000.json')
            check(diagnostic['structure_passed'], sid+'/'+arm+'/late_structure')
            check(sha(OUT/sid/arm/'mechanism_late/00040000000.npz')==diagnostic['raw_sha256'],
                  sid+'/'+arm+'/late_sha')
            for folder in ['confirm', 'monitor']:
                for path in sorted((OUT/sid/arm/folder).glob('*.npz')):
                    with np.load(path, allow_pickle=False) as f:
                        check(str(f['seed_id'])==sid and str(f['arm'])==arm and
                              str(f['code_manifest_sha256'])==identity, str(path)+'/metadata')
                        for task in tasks:
                            acc,loss = [np.asarray(f[task+'/'+key]) for key in ['acc','loss']]
                            check(acc.shape==loss.shape==(5000,) and np.isfinite(loss).all(), str(path)+'/'+task+'/shape')
                            check(np.all((acc==0)|(acc==1)), str(path)+'/'+task+'/binary')
                            check(np.array_equal(f[task+'/target'],f[task+'/labels'][:,-1]), str(path)+'/'+task+'/targets')
                            check(np.array_equal(acc,f[task+'/full_pred']==f[task+'/target']),str(path)+'/'+task+'/prediction')
                            if folder=='confirm':
                                check(np.array_equal(f[task+'/identities'],pools[task][0]) and
                                      np.array_equal(f[task+'/labels'],pools[task][1]),str(path)+'/'+task+'/fixed_pool')
                            logits=np.asarray(f[task+'/audit_logits'],np.float64)
                            ar=f[task+'/audit_rows'];labels=f[task+'/target'][ar]
                            maximum=logits.max(axis=1)
                            ce=maximum+np.log(np.exp(logits-maximum[:,None]).sum(axis=1))-logits[np.arange(len(labels)),labels]
                            check(np.max(np.abs(ce-loss[ar]))<=settings['numeric_eval_mean_ce_max_abs'],str(path)+'/'+task+'/audit_ce')
                            check(np.array_equal(logits.argmax(axis=1),f[task+'/full_pred'][ar]),str(path)+'/'+task+'/audit_prediction')
            for budget in [10000000,20000000,30000000,40000000]:
                arrays={t+'/'+m:[] for t in tasks for m in ['acc','ce']}
                for stage in range(budget-1000000,budget+1,100000):
                    with np.load(OUT/sid/arm/'confirm'/f'{stage:011d}.npz',allow_pickle=False) as f:
                        check(int(f['stage'])==stage, f'{sid}/{arm}/{stage}/stage')
                        for task in tasks:
                            for metric,field in [('acc','acc'),('ce','loss')]:
                                arrays[task+'/'+metric].append(np.asarray(f[task+'/'+field],np.float64))
                saved_rows[sid,arm,budget]={key:float((np.sum(values,axis=0)/11).mean()) for key,values in arrays.items()}
            for stage in range(10000000,40000001,100000):
                rec=read(OUT/sid/arm/'states'/f'{stage:011d}.json')
                check(rec['iter']==rec['evaluated_stage']==stage and rec['adam_count']==stage//32 and
                      rec['protection_offset']==(stage-4400000)//32,f'{sid}/{arm}/{stage}/state')
                check(rec['code_manifest_sha256']==identity,f'{sid}/{arm}/{stage}/code')
                if arm=='P25':
                    other=read(OUT/sid/'A/states'/f'{stage:011d}.json')
                    check(rec['prng']==other['prng'],f'{sid}/{stage}/matched_3_PRNG')
                retain=(stage%1000000==0 or any(stage in range(b-1000000,b+1,100000)
                    for b in [10000000,20000000,30000000,40000000]))
                if retain:
                    check(sha(OUT/rec['checkpoint_path'])==rec['checkpoint_sha256'],f'{sid}/{arm}/{stage}/checkpoint_sha')
        for path in (OUT/sid/'update_audit').glob('*.npz'):
            with np.load(path,allow_pickle=False) as f:
                u,g,p=[np.asarray(f[k],np.float64) for k in ['raw','gradient','projected']]
                dot,norm=float(g@u),float(g@g)
                expected=u-max(0.,dot-.25*max(0.,dot))/norm*g if norm>0 else u
                check(np.max(np.abs(expected-p))<=settings['numeric_projection_reference_max_abs'],str(path)+'/projection')
                check(np.linalg.norm(expected-p)/max(np.linalg.norm(u),1e-30)<=settings['numeric_projection_reference_max_relative'],str(path)+'/projection_relative')
    data=read(OUT/'analysis.json')
    frozen=read(ORIGINAL/'outputs/analysis.json')
    lookup={(r['seed'],r['arm'],r['budget']):r['metrics'] for r in frozen['per_seed']}
    lookup.update(saved_rows)
    check(len(data['per_seed'])==40 and len({(r['seed'],r['arm'],r['budget']) for r in data['per_seed']})==40,'all_40_seed_arm_budgets')
    check(data['complete'] and data['analysis_scope']=='supplementary_descriptive_only_after_original_results','scope')
    check(data['original_primary_results_unchanged']==frozen['primary'],'original_tests_unchanged')
    for row in data['per_seed']:
        expected=lookup[row['seed'],row['arm'],row['budget']]
        check(all(abs(row['metrics'][k]-v)<2e-12 for k,v in expected.items()),str((row['seed'],row['arm'],row['budget']))+'/means')
    keys=list(next(iter(lookup.values())))
    for budget in [10000000,20000000,30000000,40000000]:
        for key in keys:
            av=[lookup[s,'A',budget][key] for s in ['s18','s19','s20','s21','s22']]
            pv=[lookup[s,'P25',budget][key] for s in ['s18','s19','s20','s21','s22']]
            for arm,values in [('A',av),('P25',pv),('difference',[p-a for a,p in zip(av,pv)])]:
                verify_summary(values,data['budgets'][str(budget)]['metrics'][key][arm],f'{budget}/{key}/{arm}')
    for key in keys:
        values=[lookup[s,'P25',40000000][key]-lookup[s,'P25',10000000][key] for s in ['s18','s19','s20','s21','s22']]
        verify_summary(values,data['P25_40M_minus_10M'][key],'persistence/'+key)
    check('flip_icl' not in (OUT/'report.md').read_text(),'reported_tasks_only')
    return dict(passed=not FAILURES,checks=CHECKS,failures=FAILURES,independent_reader=True,
        code_manifest_sha256=identity,supplement_manifest_sha256=supplement_identity,
        analysis_sha256=sha(OUT/'analysis.json'),
        coverage='Four continuations/eight real exits, saved task predictions/audit logits, fixed windows, origin and retained checkpoint hashes, full state counters/matched PRNG, projection audits and five-seed descriptive statistics; no independent retraining/full-model forward.')


if __name__=='__main__':
    try:
        result=verify()
    except Exception:
        result=dict(passed=False,checks=CHECKS,failures=FAILURES+[traceback.format_exc()],independent_reader=True)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SUPPLEMENT_VERIFIED' if result['passed'] else 'SUPPLEMENT_VERIFICATION_FAILED',result['checks'])
    raise SystemExit(0 if result['passed'] else 1)
