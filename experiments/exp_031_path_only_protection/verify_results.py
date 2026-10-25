"""Separate NumPy readback: raw row/cache identities and independent statistics."""
from pathlib import Path
from datetime import datetime,timezone
import json,traceback
import numpy as np
from reference_reader import (Reader, verify_identity, feature_lookup, verify_datasets,
    verify_prerequisite_gates, verify_outputs, verify_metric_group, clustered,
    gain_test, adjust_holm, independent_ni, digest_file, is_sha, TASKS, SCALARS, STAGES)

WORK=Path(__file__).resolve().parent

def verify():
    r=Reader(WORK);methods={};independent={}
    try:
        settings=json.loads((WORK/'settings.json').read_text())
        r.require('scope/one_new_path_only_arm',settings['arms']==['P25'] and settings['ordinary_task_loss_constraints'] is False and settings['relaxation']=={'P25':.25})
        provenance,inputs,source,code=verify_identity(r)
        for n,h in settings['reference_receipt_sha256'].items():r.file('reference_receipt/'+n,WORK/'references'/n,h)
        v=json.loads((WORK/'references/verification.json').read_text());c=json.loads((WORK/'references/completion.json').read_text())
        r.require('reference/successful_readback',v['passed'] is True and not v['failures'] and c['passed'] is True)
        r.require('reference/completion_verification_bound',c['verification_sha256']==settings['reference_receipt_sha256']['verification.json'])
        r.require('reference/same_start_model',inputs.get(str((r.assets_root/'checkpoints/00004400000.eqx').resolve()))==json.loads((WORK/'references/mechanism_confirmation.json').read_text())['checkpoint_sha256'])
        labels,query=verify_datasets(r,settings,feature_lookup(r,inputs))
        verify_prerequisite_gates(r,settings,inputs,source,code)
        pre=json.loads((r.out/'preflight.json').read_text())
        r.require('preflight/single_protection_gradient',pre['checks'].get('P_constraint_count')==1)
        r.require('preflight/author_Adam_exact',pre['checks'].get('P_adam_state_exact') is True)
        new,receipts=verify_outputs(r,settings,labels,inputs,source,code)
        r.require('mechanism/passed',receipts['s5']['mechanism_passed'] is True)
        complete=receipts['s5']['completions']['P25']
        late=r.out/'s5/mechanism_late/P25_00010000000.npz'
        r.require('late/completion_state',complete.get('late_mechanism_state')=='completed')
        r.require('late/SHA_registered',is_sha(complete.get('late_mechanism_sha256')))
        r.file('late/completion_bound_SHA',late,complete.get('late_mechanism_sha256'))
        r.require('late/same_selected_and_protected_head',complete.get('late_candidate_head')==complete.get('late_protected_head')==settings['candidate_path_head']==0)
        r.require('late/structure_passed_in_completion',complete.get('late_structure_passed') is True)
        data=new['s5'];parent=Path(settings['reference_work']);old=json.loads((WORK/'references/settings.json').read_text())
        for key in ['training_seeds','start_stage','end_stage','n','confirm_stages','late_window','train_bs','protection_bs_per_task','data_sets','preserved_data_sha256']:
            r.require('reference/design/'+key,settings[key]==old[key])
        refdata=parent/'outputs/data/confirm.h5';r.file('reference/same_confirm_rows',refdata,digest_file(r.out/'data/confirm.h5'))
        for arm in settings['reference_arms']:
            metrics={t:{m:[] for m in SCALARS} for t in TASKS}
            for stage in STAGES:
                p=parent/'outputs/s5/confirm'/f'{arm}_{stage:011d}.npz';h=v['verified_file_sha256'].get(str(p.resolve()))
                r.require('reference/raw_registered/'+p.name,h is not None)
                r.file('reference/raw/'+p.name,p,h)
                with np.load(p,allow_pickle=False) as z:
                    for task in TASKS:
                        vals=verify_metric_group(r,z,task,labels[task],settings['n'],f'reference/raw/{arm}/{stage}/{task}')
                        for m,vals0 in vals.items():metrics[task][m].append(vals0)
            data[arm]={t:{m:np.sum(np.asarray(vals),axis=0,dtype=np.float64)/len(STAGES) for m,vals in ms.items()} for t,ms in metrics.items()}
        saved=json.loads((r.out/'analysis.json').read_text())
        r.require('analysis/code_manifest_bound',saved['code_manifest_sha256']==code)
        r.require('analysis/protocol_bound',saved['protocol_sha256']==digest_file(WORK/'protocol.md'))
        r.require('analysis/settings_bound',saved['settings_sha256']==digest_file(WORK/'settings.json'))
        r.require('analysis/confirm_bound',saved['confirm_data_sha256']==digest_file(r.out/'data/confirm.h5'))
        for arm,ts in data.items():
            for task,ms in ts.items():
                for m,vals in ms.items():r.subset(f'analysis/descriptive/{arm}/{task}/{m}',saved['descriptive'][arm][task][m],clustered(vals,query[task]))
        deltas={}
        for arm in settings['reference_arms']:
            deltas['P25-'+arm]={t:{m:clustered(data['P25'][t][m]-data[arm][t][m],query[t]) for m in SCALARS} for t in TASKS}
            r.subset('analysis/difference/P25-'+arm,saved['paired_comparisons']['P25-'+arm],deltas['P25-'+arm])
        gains=adjust_holm([gain_test(deltas['P25-A'][t]['acc'],'P25-A/'+t+'/acc',settings['min_icl_gain']) for t in ['icl','flip_icl']],settings['alpha'])
        r.subset('analysis/primary_Holm2',saved['primary_Holm2'],gains)
        gates={}
        for ref,tasks in [('A',['iwl_copy_avail','train_eval']),('C25',list(TASKS))]:
            tests={t+'/'+m:independent_ni(deltas['P25-'+ref][t][m],m,settings) for t in tasks for m in ['acc','loss']}
            gates['P25-'+ref]={'tests':tests,'all_passed':all(z['passed'] for z in tests.values())}
            r.subset('analysis/noninferiority/P25-'+ref,saved['noninferiority']['P25-'+ref],gates['P25-'+ref])
        methods={'path_only_tradeoff':{'gains_passed':gains['all_passed'],'noninferiority_passed':gates['P25-A']['all_passed'],'statistical_supported':gains['all_passed'] and gates['P25-A']['all_passed']},'path_only_relative_C25':{'noninferiority_passed':gates['P25-C25']['all_passed'],'statistical_supported':gates['P25-C25']['all_passed'],'strict_equivalence_tested':False}}
        r.subset('analysis/methods',saved['methods'],methods)
        independent={'primary_Holm2':gains,'noninferiority':gates,'methods':methods}
        # All five saved updates must contain exactly one protected function.
        audits=list((r.out/'s5/update_audit').glob('P25_*.json'))
        r.require('update_audit/exact_five_diagnostics',len(audits)==5)
        for p in audits:
            r.file('update_audit/file/'+p.name,p);a=json.loads(p.read_text())
            r.require('update_audit/code/'+p.name,a['code_manifest_sha256']==code and a['head']==0 and a['kind']=='P')
            for batch,z in a['results'].items():
                for key,vals in z.items():r.require('update_audit/single_path/'+p.name+'/'+batch+'/'+key,len(vals)==1 and np.isfinite(vals).all())
    except Exception as e:r.require('reader/no_exception',False,{'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()})
    passed=not r.failures
    result={'passed':passed,'status':'verified' if passed else 'failed','checks':r.checks,'failures':r.failures,'counts':dict(r.counts),'maximum_errors':r.maximum_errors,'verified_file_sha256':r.hashes,'independent_statistics':independent,'independent_late_functional':r.late_functional,'scientific_support':{k:bool(passed and z['statistical_supported']) for k,z in methods.items()},'scope':'Independent NumPy raw/cache readback and statistical recalculation; no independent full model rerun','completed_at_utc':datetime.now(timezone.utc).isoformat()}
    (r.out/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'passed':passed,'checks':r.checks,'failures':r.failures[:3],'scientific_support':result['scientific_support']},ensure_ascii=False))
    return passed

if __name__=='__main__':raise SystemExit(0 if verify() else 1)
