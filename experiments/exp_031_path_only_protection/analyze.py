"""One new path-only arm with immutable, matched exp030 reference rows."""
from pathlib import Path
import json
import numpy as np
import h5py
from reference_analysis import TASKS, METRICS, class_stats, greater, holm, ni, sha256

WORK=Path(__file__).resolve().parent

def read_window(work, arm, settings, verifier=None):
    rows={t:{m:[] for m in METRICS} for t in TASKS}
    for stage in settings['confirm_stages']:
        p=work/'outputs/s5/confirm'/f'{arm}_{stage:011d}.npz'
        if verifier is not None:
            key=str(p.resolve());expected=verifier['verified_file_sha256'].get(key)
            if expected is None or sha256(p)!=expected:
                raise ValueError('Reference result not bound to successful exp030 readback: '+str(p))
        with np.load(p,allow_pickle=False) as z:
            for t in TASKS:
                for m in METRICS:
                    v=np.asarray(z[t+'/'+m],np.float64)
                    if v.shape!=(settings['n'],) or not np.isfinite(v).all():
                        raise ValueError('Incomplete/nonfinite fixed window row: '+str(p))
                    rows[t][m].append(v)
    return {t:{m:np.mean(np.stack(v),axis=0,dtype=np.float64) for m,v in ms.items()} for t,ms in rows.items()}

def analyze():
    s=json.loads((WORK/'settings.json').read_text());v=json.loads((WORK/'references/verification.json').read_text())
    assert v['passed'] and not v['failures']
    for n,h in s['reference_receipt_sha256'].items():
        assert sha256(WORK/'references'/n)==h
    assert s['arms']==['P25'] and s['ordinary_task_loss_constraints'] is False
    old=Path(s['reference_work']);old_settings=json.loads((WORK/'references/settings.json').read_text())
    for k in ['training_seeds','start_stage','end_stage','n','confirm_stages','late_window','train_bs','protection_bs_per_task','data_sets','preserved_data_sha256']:
        assert s[k]==old_settings[k],k
    with h5py.File(WORK/'outputs/data/confirm.h5','r') as f:
        classes={t:np.asarray(f[t+'/identities'][:, -1,0]) for t in TASKS}
    data={'P25':read_window(WORK,'P25',s)}
    data.update({arm:read_window(old,arm,s,v) for arm in s['reference_arms']})
    description={arm:{t:{m:class_stats(vals,classes[t],s['alpha']) for m,vals in ms.items()} for t,ms in ts.items()} for arm,ts in data.items()}
    comparisons={}
    for arm in s['reference_arms']:
        comparisons['P25-'+arm]={t:{m:class_stats(data['P25'][t][m]-data[arm][t][m],classes[t],s['alpha']) for m in METRICS} for t in TASKS}
    gains=holm([greater(comparisons['P25-A'][t]['acc'],s['min_icl_gain'],'P25-A/'+t+'/acc') for t in ['icl','flip_icl']],s['alpha'])
    gates={}
    for reference,tasks in [('A',['iwl_copy_avail','train_eval']),('C25',list(TASKS))]:
        tests={t+'/'+m:ni(comparisons['P25-'+reference][t][m],m,s) for t in tasks for m in ['acc','loss']}
        gates['P25-'+reference]={'tests':tests,'all_passed':all(z['passed'] for z in tests.values())}
    methods={'path_only_tradeoff':{'gains_passed':gains['all_passed'],'noninferiority_passed':gates['P25-A']['all_passed'],'statistical_supported':gains['all_passed'] and gates['P25-A']['all_passed']},'path_only_relative_C25':{'noninferiority_passed':gates['P25-C25']['all_passed'],'statistical_supported':gates['P25-C25']['all_passed'],'strict_equivalence_tested':False}}
    result={'experiment':'exp031','analysis_scope':'single_seed_path_only_extension','status':'analysis_complete_pending_independent_readback','code_manifest_sha256':sha256(WORK/'code_manifest.json'),'settings_sha256':sha256(WORK/'settings.json'),'protocol_sha256':sha256(WORK/'protocol.md'),'confirm_data_sha256':sha256(WORK/'outputs/data/confirm.h5'),'window':s['late_window'],'stages':s['confirm_stages'],'n_per_task':s['n'],'descriptive':description,'paired_comparisons':comparisons,'primary_Holm2':gains,'noninferiority':gates,'methods':methods,'cross_seed_confirmation_performed':False,'reference_receipt_sha256':s['reference_receipt_sha256'],'claim_boundary':'One new fixed-seed trajectory; reference rows from verified exp030. Noninferiority is not strict equivalence; functional evidence does not establish necessity or uniqueness.'}
    (WORK/'outputs/analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    lines=['# exp031：单独路径保护结果','', '统计结果待独立读取器核验。单init5/train2、rho=.25、固定9–10M十一点；仅新增P25训练。','', '| 组 | ICL | Flip | CIWL | 原任务 |','|---|---:|---:|---:|---:|']
    for arm in ['A','B','C25','D25','R25','P25']:
        lines.append('| '+arm+' | '+' | '.join(f"{100*description[arm][t]['acc']['mean']:.3f}%" for t in ['icl','flip_icl','iwl_copy_avail','train_eval'])+' |')
    lines+=['', '联合判据：'+json.dumps(methods,ensure_ascii=False),'','完整逐题配对区间、CE、辅助指标见analysis.json；功能诊断见verification.json。结论限于该种子与窗口。']
    (WORK/'outputs/report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'analysis_complete':True,'methods':methods}))

if __name__=='__main__':analyze()
