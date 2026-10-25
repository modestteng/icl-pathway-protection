"""Independent CPU reader: no worker, JAX, Equinox, Torch or model import."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import math
import struct
import time
import numpy as np
import h5py
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
CFG = json.loads((ROOT / 'settings.json').read_text())
C, K = CFG['classes'], CFG['query_per_class']
NQ, NI, H = C*K, C*K*2, K//2
assert K == CFG['support_per_class'] and K%2 == 0


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def raw_source(split):
    # Deliberately independent from dataset_io.py; no worker imports.
    if CFG['dataset_kind'] == 'k49':
        imfile = ROOT / 'raw' / ('k49-' + split + '-imgs.npz')
        lbfile = ROOT / 'raw' / ('k49-' + split + '-labels.npz')
        with np.load(imfile, allow_pickle=False) as z:
            images = np.array(z['arr_0'])
        with np.load(lbfile, allow_pickle=False) as z:
            labels = np.array(z['arr_0'])
        return images, labels
    import tarfile
    with tarfile.open(ROOT / 'raw/cifar-100-binary.tar.gz', 'r:gz') as f:
        payload = f.extractfile('cifar-100-binary/' + split + '.bin').read()
    n = CFG['official_sizes'][split]
    assert len(payload) == n * 3074
    records = np.frombuffer(payload, dtype='u1').reshape(n, 3074)
    channels = [records[:, 2 + k*1024:2 + (k+1)*1024].reshape(n, 32, 32) for k in range(3)]
    return np.stack(channels, axis=-1), records[:, 1].copy()


def interval(values):
    values = np.asarray(values, np.float64)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(C))
    r = float(t.ppf(.975, (C-1))) * se
    return dict(mean=mean, se=se, lower95=mean-r, upper95=mean+r,
                clusters=C, df=(C-1), status='t_query_class' if se else 'degenerate')


def main():
    started = time.monotonic()
    checks, errors = 0, {}
    def check(condition, name):
        nonlocal checks
        checks += 1
        if not bool(condition):
            raise AssertionError(name)
    code = json.loads((ROOT / 'code_manifest.json').read_text())
    for name, expected in code.items():
        check(digest(ROOT / name) == expected, 'sealed source '+name)
    for name, expected in CFG['downloads'].items():
        check(digest(ROOT/'raw'/name)==expected, 'raw source SHA '+name)
    dm = json.loads((OUT / 'data_manifest.json').read_text())
    enc = json.loads((OUT / 'encoding_receipt.json').read_text())
    check(digest(OUT / 'selected_images.h5') == dm['selected_images_sha256'], 'image HDF')
    check(digest(OUT / 'episodes.npz') == dm['episodes_sha256'], 'episode identity')
    check(digest(OUT / 'features.h5') == enc['features_sha256'], 'feature identity')
    check(enc['parameters_before'] == enc['parameters_after'], 'frozen encoder')
    check(enc['weights_sha256'] == enc['original_weights_sha256'], 'original encoder weights')
    with h5py.File(OUT / 'selected_images.h5', 'r') as f:
        images, cls = f['images'][:], f['classes'][:]
        splits, original = f['official_split'][:], f['original_index'][:]
        hashes = f['pixel_sha256'][:]
    check(images.shape == (NI,*CFG['image_shape']), 'image count')
    check(len(set(hashes)) == NI, 'all raw selected pixels distinct')
    check(np.array_equal(splits, np.repeat([0,1],NQ)), 'disjoint official roles')
    for split, role in [('train',0),('test',1)]:
        ix = np.flatnonzero(splits == role)
        raw, labs = raw_source(split)
        check(np.array_equal(raw[original[ix]], images[ix]), 'raw selected pixels '+split)
        check(np.array_equal(labs[original[ix]], cls[ix]), 'raw selected labels '+split)
        check(np.array_equal(np.bincount(cls[ix],minlength=C),np.full(C,K)), 'per-class role '+split)
        for j in ix:
            check(hashlib.sha256(images[j].tobytes()).hexdigest().encode() == hashes[j], 'pixel hash')
        del raw, labs
    d = np.load(OUT / 'episodes.npz')
    ids, ys, qc = d['image_ids'], d['labels'], d['query_classes']
    check(ids.shape == (NQ,3) and ys.shape == (NQ,3), 'episode sizes')
    check(np.array_equal(ids[:,-1],np.arange(NQ,NI)), 'one query per image')
    check((ids[:,:2] < NQ).all(), 'support role')
    check(np.array_equal(np.sort(ys[:,:2],axis=1),np.tile([0,1],(NQ,1))), 'candidate tags')
    ec = cls[ids]
    match = ec[:,:2] == ec[:,-1:]
    check((match.sum(1)==1).all(), 'one matching exemplar')
    check(np.array_equal(ys[:,-1],np.sum(ys[:,:2]*match,axis=1)), 'context answer')
    check(np.array_equal(qc,ec[:,-1]), 'query class identity')
    for c in range(C):
        ix = qc == c
        check(ix.sum() == K, 'queries per class')
        check(np.bincount(ys[ix,-1],minlength=2).tolist() == [H,H], 'balanced targets')
        check(match[ix].sum(0).tolist() == [H,H], 'balanced order')
    with h5py.File(OUT / 'features.h5','r') as f:
        check(f['features'].shape == (NI,512), 'feature shape')
        check(np.isfinite(f['features'][:]).all(), 'finite features')
    summaries, rows = {}, {}
    names = json.loads((OUT/'class_names.json').read_text())
    check(len(names)==C, 'balanced class mapping')
    for arm in ['A','P25']:
        receipt = json.loads((OUT / (arm+'_evaluation_receipt.json')).read_text())
        r = dict(np.load(OUT / (arm+'_rows.npz')))
        check(receipt['checkpoint_sha256'] == CFG['models'][arm]['sha256'], 'checkpoint binding')
        check(digest(CFG['models'][arm]['path']) == receipt['checkpoint_sha256'], 'unchanged checkpoint')
        check(receipt['stage']==10000000 and receipt['rows']==NQ, 'stage and rows')
        check(receipt['episodes_sha256']==dm['episodes_sha256'], 'same episodes '+arm)
        check(receipt['features_sha256']==enc['features_sha256'], 'same features '+arm)
        check(digest(OUT/(arm+'_rows.npz'))==receipt['rows_sha256'], 'row file identity')
        check(np.array_equal(r['image_ids'],ids) and np.array_equal(r['target'],ys[:,-1]), 'stored row order')
        maxerr = dict(ce=0., pair_ce=0., context_mass=0.)
        with h5py.File(OUT/(arm+'_logits.h5'),'r') as f:
            check(f['logits'].shape==(NQ,12800), 'all logits shape')
            for lo in range(0,NQ,64):
                hi=min(lo+64,NQ)
                z=f['logits'][lo:hi].astype(np.float64)
                target=ys[lo:hi,-1]
                check(np.isfinite(z).all(), 'finite logits')
                mx=z.max(1)
                logsum=mx+np.log(np.exp(z-mx[:,None]).sum(1))
                ce=logsum-z[np.arange(hi-lo),target]
                pm=z[:,:2].max(1)
                pz=pm+np.log(np.exp(z[:,:2]-pm[:,None]).sum(1))
                pp=z[:,:2].argmax(1)
                check(np.array_equal(z.argmax(1),r['full_pred'][lo:hi]), 'full argmax')
                check(np.array_equal(pp,r['pair_pred'][lo:hi]), 'pair argmax')
                for key,value in [('ce',ce),('pair_ce',pz-z[np.arange(hi-lo),target]),
                                  ('context_mass',np.exp(pz-logsum))]:
                    error=float(np.abs(value-r[key][lo:hi]).max())
                    maxerr[key]=max(maxerr[key],error)
                    check(error <= (3e-5 if key=='ce' else 1e-10), 'independent '+key)
        check(digest(OUT/(arm+'_logits.h5'))==receipt['logits_sha256'], 'complete logits hash')
        rows[arm]=dict(acc=(r['full_pred']==r['target']).astype(float),
                       ce=r['ce'].astype(float), pair_acc=(r['pair_pred']==r['target']).astype(float))
        summaries[arm]={k:interval([v[qc==c].mean() for c in range(C)]) for k,v in rows[arm].items()}
        errors[arm]=maxerr
    differences={k:interval([(rows['P25'][k]-rows['A'][k])[qc==c].mean() for c in range(C)])
                 for k in ['acc','ce','pair_acc']}
    transitions=dict(A_wrong_P25_correct=int(((rows['A']['acc']==0)&(rows['P25']['acc']==1)).sum()),
                     A_correct_P25_wrong=int(((rows['A']['acc']==1)&(rows['P25']['acc']==0)).sum()),
                     both_correct=int(((rows['A']['acc']==1)&(rows['P25']['acc']==1)).sum()),
                     both_wrong=int(((rows['A']['acc']==0)&(rows['P25']['acc']==0)).sum()))
    check(sum(transitions.values())==NQ, 'paired transitions total')
    check(abs(differences['acc']['mean']-(transitions['A_wrong_P25_correct']-
          transitions['A_correct_P25_wrong'])/NQ)<1e-12, 'paired effect identity')
    with (OUT/'per_class.csv').open('w',newline='') as f:
        fields=['class_id','character','queries','A_acc','P25_acc','P25_minus_A_acc','A_ce','P25_ce','P25_minus_A_ce']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for c in range(C):
            ix=qc==c
            w.writerow(dict(class_id=c,character=names[c],queries=K,
                A_acc=rows['A']['acc'][ix].mean(),P25_acc=rows['P25']['acc'][ix].mean(),
                P25_minus_A_acc=(rows['P25']['acc']-rows['A']['acc'])[ix].mean(),
                A_ce=rows['A']['ce'][ix].mean(),P25_ce=rows['P25']['ce'][ix].mean(),
                P25_minus_A_ce=(rows['P25']['ce']-rows['A']['ce'])[ix].mean()))
    primary=differences['acc']
    standing=('positive_difference' if primary['status']!='degenerate' and primary['lower95']>0 else
              'negative_difference' if primary['status']!='degenerate' and primary['upper95']<0 else
              'difference_not_resolved')
    analysis=dict(experiment=CFG['experiment_id'],stage=10000000,models=summaries,
                  P25_minus_A=differences,paired_full_accuracy_counts=transitions,
                  primary_standing=standing,n_queries=NQ,n_classes=C,training_trajectories=1,
                  interval_scope=CFG['confidence_interval'],
                  limitation='Fixed support pools and contexts; cross-class distractor/support dependence is not fully modeled by one-way query-class clustering. External source is unseen to these Transformer training runs; original encoder pretraining is unchanged and no pretraining-disjointness claim is made.')
    save('analysis.json',analysis)
    save('verification.json',dict(passed=True,checks=checks,failures=[],independent_reader=True,
         coverage=f'All {NI} raw image identities; all {NQ} episode structures; all {NQ}x12800 saved logits per model; no independent whole-encoder rerun.',
         max_metric_errors=errors,analysis_seconds=time.monotonic()-started,
         source_manifest_sha256=digest(ROOT/'code_manifest.json')))
    analysis['critical_t95'] = float(t.ppf(.975,C-1))
    analysis['class_directions'] = dict(positive=sum((rows['P25']['acc']-rows['A']['acc'])[qc==c].mean()>0 for c in range(C)), equal=sum((rows['P25']['acc']-rows['A']['acc'])[qc==c].mean()==0 for c in range(C)), negative=sum((rows['P25']['acc']-rows['A']['acc'])[qc==c].mean()<0 for c in range(C)))
    analysis['class_directions'] = {k:int(v) for k,v in analysis['class_directions'].items()}
    save('analysis.json',analysis)
    lines=['# '+CFG['dataset_name']+'：冻结 A/P25 的跨数据来源评价','',
      f"{C} 类，每类 {K} 支持和 {K} 查询；共 {NI} 图像、{NQ} 题。原编码器与同一组10M模型保持冻结。",'',
      '| 指标 | A | P25 | P25−A [95%区间] |','|---|---:|---:|---:|']
    for key,title,scale,unit in [('acc','完整12800标签准确率',100,'% / pp'),('ce','完整交叉熵',1,'nat'),('pair_acc','二候选准确率（辅助）',100,'% / pp')]:
      a,b,di=summaries['A'][key]['mean'],summaries['P25'][key]['mean'],differences[key]
      lines.append(f"| {title} | {a*scale:.4f} | {b*scale:.4f} | {di['mean']*scale:+.4f} [{di['lower95']*scale:+.4f}, {di['upper95']*scale:+.4f}] {unit} |")
    lines += ['', '主准确率差异：'+standing+'。', '', '按查询类别聚类的配对区间；固定支持池/上下文及单条训练轨迹。共享支持和干扰样本的跨类依赖未由单向聚类完全覆盖。', '', '配对计数：'+json.dumps(transitions,ensure_ascii=False), '', '各类方向：'+json.dumps(analysis['class_directions'],ensure_ascii=False), '', '图像适配：'+CFG['image_adapter'], '', '原始数据、题目、特征、两组全部logits和指标已由独立CPU读取器核对。完整logits保留Windows，紧凑产物取回Mac。', '', '本报告由AI辅助执行与汇总；没有在目标数据上更新任何参数。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(standing=standing,models={a:{k:v['mean'] for k,v in m.items()} for a,m in summaries.items()}, differences=differences,checks=checks)),flush=True)


if __name__=='__main__':
    main()
