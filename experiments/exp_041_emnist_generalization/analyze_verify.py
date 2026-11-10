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


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def raw_idx(path):
    with gzip.open(path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        nd = magic % 256
        assert magic // 256 == 8
        shape = struct.unpack('>'+'I'*nd, f.read(4*nd))
        a = np.frombuffer(f.read(), np.uint8).reshape(shape).copy()
    return a


def interval(values):
    values = np.asarray(values, np.float64)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(47))
    r = float(t.ppf(.975, 46)) * se
    return dict(mean=mean, se=se, lower95=mean-r, upper95=mean+r,
                clusters=47, df=46, status='t_query_class' if se else 'degenerate')


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
    check(images.shape == (4700,28,28), 'image count')
    check(len(set(hashes)) == 4700, 'all raw selected pixels distinct')
    check(np.array_equal(splits, np.repeat([0,1],2350)), 'disjoint official roles')
    for split, role in [('train',0),('test',1)]:
        ix = np.flatnonzero(splits == role)
        raw = raw_idx(ROOT / 'raw' / ('emnist-balanced-'+split+'-images-idx3-ubyte.gz'))
        labs = raw_idx(ROOT / 'raw' / ('emnist-balanced-'+split+'-labels-idx1-ubyte.gz'))
        check(np.array_equal(raw[original[ix]], images[ix]), 'raw selected pixels '+split)
        check(np.array_equal(labs[original[ix]], cls[ix]), 'raw selected labels '+split)
        check(np.array_equal(np.bincount(cls[ix],minlength=47),np.full(47,50)), 'per-class role '+split)
        for j in ix:
            check(hashlib.sha256(images[j].tobytes()).hexdigest().encode() == hashes[j], 'pixel hash')
        del raw, labs
    d = np.load(OUT / 'episodes.npz')
    ids, ys, qc = d['image_ids'], d['labels'], d['query_classes']
    check(ids.shape == (2350,3) and ys.shape == (2350,3), 'episode sizes')
    check(np.array_equal(ids[:,-1],np.arange(2350,4700)), 'one query per image')
    check((ids[:,:2] < 2350).all(), 'support role')
    check(np.array_equal(np.sort(ys[:,:2],axis=1),np.tile([0,1],(2350,1))), 'candidate tags')
    ec = cls[ids]
    match = ec[:,:2] == ec[:,-1:]
    check((match.sum(1)==1).all(), 'one matching exemplar')
    check(np.array_equal(ys[:,-1],np.sum(ys[:,:2]*match,axis=1)), 'context answer')
    check(np.array_equal(qc,ec[:,-1]), 'query class identity')
    for c in range(47):
        ix = qc == c
        check(ix.sum() == 50, 'queries per class')
        check(np.bincount(ys[ix,-1],minlength=2).tolist() == [25,25], 'balanced targets')
        check(match[ix].sum(0).tolist() == [25,25], 'balanced order')
    with h5py.File(OUT / 'features.h5','r') as f:
        check(f['features'].shape == (4700,512), 'feature shape')
        check(np.isfinite(f['features'][:]).all(), 'finite features')
    summaries, rows = {}, {}
    names = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt')
    check(len(names)==47, 'balanced class mapping')
    for arm in ['A','P25']:
        receipt = json.loads((OUT / (arm+'_evaluation_receipt.json')).read_text())
        r = dict(np.load(OUT / (arm+'_rows.npz')))
        check(receipt['checkpoint_sha256'] == CFG['models'][arm]['sha256'], 'checkpoint binding')
        check(digest(CFG['models'][arm]['path']) == receipt['checkpoint_sha256'], 'unchanged checkpoint')
        check(receipt['stage']==10000000 and receipt['rows']==2350, 'stage and rows')
        check(receipt['episodes_sha256']==dm['episodes_sha256'], 'same episodes '+arm)
        check(receipt['features_sha256']==enc['features_sha256'], 'same features '+arm)
        check(digest(OUT/(arm+'_rows.npz'))==receipt['rows_sha256'], 'row file identity')
        check(np.array_equal(r['image_ids'],ids) and np.array_equal(r['target'],ys[:,-1]), 'stored row order')
        maxerr = dict(ce=0., pair_ce=0., context_mass=0.)
        with h5py.File(OUT/(arm+'_logits.h5'),'r') as f:
            check(f['logits'].shape==(2350,12800), 'all logits shape')
            for lo in range(0,2350,64):
                hi=min(lo+64,2350)
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
        summaries[arm]={k:interval([v[qc==c].mean() for c in range(47)]) for k,v in rows[arm].items()}
        errors[arm]=maxerr
    differences={k:interval([(rows['P25'][k]-rows['A'][k])[qc==c].mean() for c in range(47)])
                 for k in ['acc','ce','pair_acc']}
    transitions=dict(A_wrong_P25_correct=int(((rows['A']['acc']==0)&(rows['P25']['acc']==1)).sum()),
                     A_correct_P25_wrong=int(((rows['A']['acc']==1)&(rows['P25']['acc']==0)).sum()),
                     both_correct=int(((rows['A']['acc']==1)&(rows['P25']['acc']==1)).sum()),
                     both_wrong=int(((rows['A']['acc']==0)&(rows['P25']['acc']==0)).sum()))
    check(sum(transitions.values())==2350, 'paired transitions total')
    check(abs(differences['acc']['mean']-(transitions['A_wrong_P25_correct']-
          transitions['A_correct_P25_wrong'])/2350)<1e-12, 'paired effect identity')
    with (OUT/'per_class.csv').open('w',newline='') as f:
        fields=['class_id','character','queries','A_acc','P25_acc','P25_minus_A_acc','A_ce','P25_ce','P25_minus_A_ce']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for c in range(47):
            ix=qc==c
            w.writerow(dict(class_id=c,character=names[c],queries=50,
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
                  primary_standing=standing,n_queries=2350,n_classes=47,training_trajectories=1,
                  interval_scope=CFG['confidence_interval'],
                  limitation='Fixed support pools and contexts; cross-class distractor/support dependence is not fully modeled by one-way query-class clustering. External source is unseen to these Transformer training runs, but semantic character identities may overlap Omniglot.')
    save('analysis.json',analysis)
    save('verification.json',dict(passed=True,checks=checks,failures=[],independent_reader=True,
         coverage='All 4700 raw image identities; all 2350 episode structures; all 2350x12800 saved logits per model; no independent whole-encoder rerun.',
         max_metric_errors=errors,analysis_seconds=time.monotonic()-started,
         source_manifest_sha256=digest(ROOT/'code_manifest.json')))
    lines=['# exp041：EMNIST跨数据来源普通ICL评价','',
           '## Material Passport','', '- Origin Skill: academic-research-suite / experiment-agent',
           '- Origin Mode: run','- Origin Date: 2026-10-20','- Verification Status: ANALYZED',
           '- Version Label: exp041_result_v1','',
           'Windows RTX4070，冻结主轨迹A/P25的10M检查点；47类，每类50支持与50查询，总4700图像、2350题。',
           '', '| 指标 | A | P25 | P25−A [95%区间] |','|---|---:|---:|---:|']
    for key,title,scale,unit in [('acc','完整12800标签准确率',100,'% / pp'),('ce','完整交叉熵',1,'nat'),
                                  ('pair_acc','二候选准确率（辅助）',100,'% / pp')]:
        a,b,di=summaries['A'][key]['mean'],summaries['P25'][key]['mean'],differences[key]
        lines.append(f"| {title} | {a*scale:.4f} | {b*scale:.4f} | {di['mean']*scale:+.4f} [{di['lower95']*scale:+.4f}, {di['upper95']*scale:+.4f}] {unit} |")
    lines += ['', '主准确率对比判读：'+standing+'。唯一主对比事前固定，未按结果换图、换检查点或调预处理。',
              '', '全部逐题指标由独立CPU读取器从保存logits重算通过；原始图像、题目及模型SHA核对通过。',
              '', '配对改对/改错计数：'+json.dumps(transitions,ensure_ascii=False)+'。',
              '', '区间按47个查询类别聚类，只描述单轨迹、固定数据池/上下文的评价差异；共享支持与干扰类别的依赖未被单向聚类完全覆盖。',
              '这是跨数据来源/书写样本评价，不证明全部字符语义未见，也不证明其他训练种子、任务或领域的普遍优势。',
              '', 'ResNet18编码阶段总秒数：'+str(enc['encoding_stage_seconds'])+'；其中CUDA前向秒数：'+str(enc['gpu_forward_seconds'])+'。',
              '编码峰值显存分配字节：'+str(enc['peak_cuda_allocated_bytes'])+'。',
              '', '完整logits保留Windows，取回清单中的特征、抽样身份、逐题指标、汇总、核验与日志。',
              '', 'idea → change → result → decision：检验冻结路径保护模型的跨数据来源ICL → 固定原编码器/模型，仅替换评价题 → '+standing+' → 按此评价范围保留结果。',
              '', '本报告由AI辅助执行与汇总，原始来源、冻结协议、代码与核验记录随实验保存。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(standing=standing,models={a:{k:v['mean'] for k,v in m.items()} for a,m in summaries.items()},
                         differences=differences,checks=checks)),flush=True)


if __name__=='__main__':
    main()
