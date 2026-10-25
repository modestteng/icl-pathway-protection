"""Paired, within-checkpoint activation interventions using frozen author modules."""
from pathlib import Path
from time import perf_counter
import argparse
import csv
import hashlib
import json
import platform
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--preflight-only', action='store_true')
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
args = parser.parse_args()
work = Path(__file__).resolve().parent
root = args.root
source = root / 'reproduction_support/source'
sys.path.insert(0, str(source))
import numpy as np
import h5py
import jax
import jax.numpy as jnp
import equinox as eqx
import main_utils
import main
import opto
from author_forward_metrics import make_forward_fn_ic_only

t0 = perf_counter()
out = work / 'outputs'
out.mkdir(exist_ok=True)
expected = json.loads((work / 'expected_inputs.json').read_text())
STAGES = [4400000, 8000000, 20000000]
N, BS = 5000, 128
assert expected['stages'] == STAGES
assert jax.default_backend() == 'gpu'
assert jax.config.jax_default_matmul_precision is None
jax.config.update('jax_default_matmul_precision', 'float32')
KEY = jax.random.PRNGKey(0)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


cfg = root / 'exp_002_main_reproduction/runs/main/config.json'
ev = root / 'exp_002_main_reproduction/runs/default_eval/eval_data.h5'
assert sha(cfg) == expected['configuration_sha256']
assert sha(ev) == expected['evaluation_data_sha256']
for name, digest in expected['source_sha256'].items():
    assert sha(source / name) == digest, name
assert sha(work / 'reference_learning_curves.tsv') == expected['baseline_curve_sha256']
assert sha(work / 'author_forward_metrics.py') == expected['author_forward_sha256']


def checkpoint(stage):
    return root / f'exp_002_main_reproduction/runs/main_resume_00002950016/checkpoints/{stage:011d}.eqx'


for stage in STAGES:
    assert sha(checkpoint(stage)) == expected['checkpoint_sha256_by_stage'][str(stage)]
opts = main_utils.get_opts_from_json_file(cfg)
assert opts.depth == 2 and opts.num_heads == 8 and opts.d_model == 64
assert opts.mlp_ratio is None and opts.pos_embedding_type == 'ape'
model0 = main_utils.get_model_from_opts(opts, (512,))
for obj in jax.tree_util.tree_leaves(model0, is_leaf=lambda x: isinstance(x, eqx.nn.Dropout)):
    if isinstance(obj, eqx.nn.Dropout):
        assert obj.p == 0
optimizer = main_utils.get_optimizer_from_opts(opts)
template = dict(iter=-1, model=model0,
                opt_state=optimizer.init(eqx.filter(model0, eqx.is_array)),
                seeds=dict(eval_model_seed=KEY, train_data_seed=KEY, train_model_seed=KEY))


def restore(stage):
    ck = eqx.tree_deserialise_leaves(checkpoint(stage), template)
    assert int(ck['iter']) == stage and int(ck['opt_state'][0].count) == stage // 32
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(ck)
               if eqx.is_array(v))
    return ck['model']


def make_conditions():
    conditions = []
    for head in [-1] + list(range(8)):
        h = 'all' if head == -1 else str(head)
        for positions in [[1], [3], [1, 3]]:
            conditions.append(dict(name=f'K_h{h}_p'+''.join(map(str, positions)),
                                   kind='K', head=head, positions=positions, reverse=False))
        for kind in ['V', 'Q', 'KV']:
            conditions.append(dict(name=f'{kind}_h{h}', kind=kind, head=head,
                                   positions=[4] if kind == 'Q' else [1, 3], reverse=False))
    conditions.append(dict(name='K_self', kind='K', head=-1, positions=[1, 3],
                           reverse=False, self_patch=True))
    for kind in ['K', 'V', 'Q', 'KV']:
        conditions.append(dict(name=f'{kind}_reverse', kind=kind, head=-1,
                               positions=[4] if kind == 'Q' else [1, 3], reverse=True))
    for sender in range(8):
        for receiver in [-1] + list(range(8)):
            h = 'all' if receiver == -1 else str(receiver)
            conditions.append(dict(name=f'L1_{sender}_to_K_h{h}', kind='edge',
                                   sender=sender, head=receiver, positions=[1, 3], reverse=False))
    conditions.append(dict(name='L1_all_to_K_all', kind='edge', sender=-1,
                           head=-1, positions=[1, 3], reverse=False))
    assert len(conditions) == 132 and len({c['name'] for c in conditions}) == 132
    return conditions


conditions = [c for c in make_conditions() if c['name'] == 'K_hall_p13' or c['name'] in {f'L1_{i}_to_K_hall' for i in range(8)}]
condition_path = work / 'conditions.json'
if condition_path.exists():
    assert json.loads(condition_path.read_text()) == conditions
else:
    condition_path.write_text(json.dumps(conditions, indent=2) + '\n')


def metrics(r, y):
    logits = r['out'][-1]
    label0, label1 = y[0], y[1]
    ci = jnp.argmax(y[:2] == y[-1])
    oi = 1 - ci
    pair = logits[y[:2]]
    margin = pair[oi] - pair[ci]
    # Match author's full masked argmax tie rule (lowest label ID), not first position.
    pred = jnp.where(pair[0] > pair[1], label0,
                    jnp.where(pair[1] > pair[0], label1, jnp.minimum(label0, label1)))
    a = r['transformer_output']['block_outputs'][1]['attn_output']
    attn = a['attn_scores'][0, :, 4, :][:, jnp.array([1, 3])]
    score = a['attn_pre_softmax'][0, :, 4, :][:, jnp.array([1, 3])]
    return dict(margin=margin, acc=(pred == y[-1]).astype(jnp.float32),
                donor_choice=(pred == y[oi]).astype(jnp.float32),
                context_prob=jax.nn.sigmoid(-margin),
                context_loss=jax.nn.softplus(margin),
                full_loss=jax.scipy.special.logsumexp(logits) - logits[y[-1]],
                full_acc=(jnp.argmax(logits) == y[-1]).astype(jnp.float32),
                attn_margin=attn[:, oi] - attn[:, ci],
                score_margin=score[:, oi] - score[:, ci])


def features(model, r, y):
    blocks = r['transformer_output']['block_outputs']
    a = blocks[1]['attn_output']
    z = blocks[0]['attn_output']['values_pre_proj']
    proj = model.transformer.blocks[0].attn.proj
    head_masks = jnp.repeat(jnp.eye(8, dtype=z.dtype), 8, axis=1)
    masked = z[None, :, :] * head_masks[:, None, :]
    # Use the unchanged full-width author projection, rather than 8D sub-GEMMs.
    contributions = jax.vmap(jax.vmap(proj))(masked)
    if proj.bias is not None:
        contributions = contributions - proj.bias
    return dict(q=a['q'], k=a['k'], v=a['v'], l1_out=blocks[0]['out'],
                l1_contrib=contributions,
                l1_projected=blocks[0]['attn_output']['values_post_proj'],
                pair_logits=r['out'][-1, y[:2]])


def native_one(model, x, y):
    r = model.call_with_all_aux(examples=x, labels=y, key=KEY)
    return metrics(r, y), features(model, r, y)


native_fn = eqx.filter_jit(jax.vmap(native_one, in_axes=(None, 0, 0)))
op = argparse.ArgumentParser()
opto.add_args_to_parser(op)
author_fn = main.make_batched_fn(make_forward_fn_ic_only(op.parse_args([])), 1000)


def masks(c):
    q = np.zeros((1, 8, 5, 8), dtype=bool)
    k, v = q.copy(), q.copy()
    heads = list(range(8)) if c['head'] == -1 else [c['head']]
    kind = c['kind']
    for h in heads:
        if kind == 'Q':
            q[0, h, 4, :] = True
        if kind in ['K', 'KV', 'edge']:
            for p in c['positions']:
                k[0, h, p, :] = True
        if kind in ['V', 'KV']:
            for p in c['positions']:
                v[0, h, p, :] = True
    sender = np.zeros(8, np.float32)
    if kind == 'edge':
        sender[:] = 1 if c['sender'] == -1 else 0
        if c['sender'] != -1:
            sender[c['sender']] = 1
    return tuple(jnp.asarray(a) for a in (q, k, v, sender))


def patched_forward(model, x, y, base, donor, qm, km, vm, sender, edge):
    def edge_key(_):
        delta = jnp.sum((donor['l1_contrib'] - base['l1_contrib']) * sender[:, None, None], axis=0)
        label_positions = jnp.array([False, True, False, True, False])[:, None]
        h = base['l1_out'] + jnp.where(label_positions, delta, 0.)
        block = model.transformer.blocks[1]
        norm = jax.vmap(block.norm1)(h)
        return block.attn.call_with_all_aux(norm, key=KEY)['k']

    dk = jax.lax.cond(edge, edge_key, lambda _: donor['k'], operand=None)
    values = dict(q=jnp.where(qm, donor['q'], base['q']),
                  k=jnp.where(km, dk, base['k']),
                  v=jnp.where(vm, donor['v'], base['v']))
    block_cache = [dict(), dict(attn_output=values)]
    block_mask = [dict(), dict(attn_output={n: jnp.ones_like(a, dtype=bool)
                                          for n, a in values.items()})]
    return model.call_with_all_aux(examples=x, labels=y, key=KEY,
              cache=dict(transformer_output=dict(block_outputs=block_cache)),
              cache_mask=dict(transformer_output=dict(block_outputs=block_mask)))


def patch_one(model, x, y, base, donor, qm, km, vm, sender, edge):
    return metrics(patched_forward(model, x, y, base, donor, qm, km, vm, sender, edge), y)


patch_fn = eqx.filter_jit(jax.vmap(patch_one,
                        in_axes=(None, 0, 0, 0, 0, None, None, None, None, None)))


def batched(fn, model, x, y, b=None, d=None, c=None):
    outputs = []
    for start in range(0, len(x), BS):
        stop = min(start + BS, len(x))
        if c is None:
            z = fn(model, x[start:stop], y[start:stop])
        else:
            bz = jax.tree_util.tree_map(lambda a: a[start:stop], b)
            dz = jax.tree_util.tree_map(lambda a: a[start:stop], d)
            z = fn(model, x[start:stop], y[start:stop], bz, dz, *masks(c),
                   jnp.asarray(c['kind'] == 'edge'))
        outputs.append(jax.tree_util.tree_map(lambda a: np.asarray(a), z))
    return jax.tree_util.tree_map(lambda *aa: np.concatenate(aa, axis=0), *outputs)


def cluster_stats(delta, labels):
    delta = np.asarray(delta, np.float64)
    labels = np.asarray(labels)
    _, inv = np.unique(labels, return_inverse=True)
    groups = inv.max() + 1
    mean = float(delta.mean())
    sums = np.bincount(inv, weights=delta - mean)
    se = float(np.sqrt(groups / (groups - 1) * np.sum(sums * sums)) / len(delta))
    return dict(mean=mean, se=se, ci95_low=mean - 1.96 * se, ci95_high=mean + 1.96 * se,
                query_classes=int(groups))


def audit_one(model, x, y, b, d, qm, km, vm, sender, edge):
    r = patched_forward(model, x, y, b, d, qm, km, vm, sender, edge)
    a = r['transformer_output']['block_outputs'][1]['attn_output']
    raw = model.call_with_all_aux(examples=x, labels=y, key=KEY)
    raw0 = raw['transformer_output']['block_outputs'][0]
    proj = model.transformer.blocks[0].attn.proj
    reconstructed = jnp.sum(b['l1_contrib'], axis=0)
    if proj.bias is not None:
        reconstructed = reconstructed + proj.bias
    calculated = jax.nn.softmax(a['attn_pre_softmax'], axis=-1)
    pred_score = (a['q'] @ jnp.swapaxes(a['k'], -1, -2)) * model.transformer.blocks[1].attn.scale
    causal = jnp.tril(jnp.ones((5, 5), bool))
    return dict(q_max=jnp.max(jnp.abs(a['q'] - b['q'])),
                v_max=jnp.max(jnp.abs(a['v'] - b['v'])),
                k_target_max=jnp.max(jnp.where(km, jnp.abs(a['k'] - d['k']), 0)),
                k_other_max=jnp.max(jnp.where(~km, jnp.abs(a['k'] - b['k']), 0)),
                upstream_max=jnp.max(jnp.abs(r['transformer_output']['block_outputs'][0]['out'] - raw0['out'])),
                attn_recompute_max=jnp.max(jnp.abs(calculated - a['attn_scores'])),
                score_recompute_max=jnp.max(jnp.where(causal, jnp.abs(pred_score-a['attn_pre_softmax']), 0)),
                causal_future_max=jnp.max(jnp.where(~causal, a['attn_scores'], 0)),
                row_sum_max=jnp.max(jnp.abs(a['attn_scores'].sum(-1)-1)),
                l1_split_max=jnp.max(jnp.abs(reconstructed-b['l1_projected'])),
                final_logit_max=jnp.max(jnp.abs(r['out'][-1,y[:2]]-b['pair_logits'])))


audit_fn = eqx.filter_jit(jax.vmap(audit_one,
                        in_axes=(None,0,0,0,0,None,None,None,None,None)))


# New orchestration only: intervention functions above are reused from exp_013.
EVALUATORS = ['icl']
FEATURE = root / 'exp_002_main_reproduction/features/omniglot_features_norotate.h5'
assert sha(FEATURE) == expected['feature_sha256']
with h5py.File(FEATURE, 'r') as f:
    feature_data = np.asarray(f['resnet18/224/feat'][:1600, :5], dtype=np.float32)
    assert np.array_equal(f['original_class'][:1600], np.arange(1600))
    assert np.all(f['augmentation'][:1600] == 0)
assert feature_data.shape == (1600, 5, 512)
lookup = {}
for cls in range(1600):
    for ex in range(5):
        key = hashlib.sha256(feature_data[cls, ex].tobytes()).digest()
        assert key not in lookup, 'Duplicate feature identity'
        lookup[key] = (cls, ex)

def identify(x):
    return np.asarray([lookup[hashlib.sha256(v.tobytes()).digest()]
                       for v in x.reshape(-1,512)], np.int32).reshape(N,3,2)

data, data_audit = {}, {}
with h5py.File(ev, 'r') as f:
    for name in EVALUATORS:
        x = np.asarray(f[name]['examples'], np.float32)
        y = np.asarray(f[name]['labels'], np.int32)
        assert x.shape == (N,3,512) and y.shape == (N,3)
        assert np.isfinite(x).all() and np.all(y[:,0] != y[:,1])
        assert np.all((y[:,:2] == y[:,-1,None]).sum(1) == 1)
        ids = identify(x)
        qc = ids[:,-1,0]
        same = ids[:,:2,0] == qc[:,None]
        if name == 'icl':
            assert np.all(same.sum(1) == 1)
            assert np.all(y[:,:2][same] == y[:,-1])
            assert np.all(np.sort(y[:,:2],axis=1) == np.array([0,1]))
        else:
            assert np.all(same.sum(1) == 0)
            assert np.all(ids[:,0,0] != ids[:,1,0])
            assert np.array_equal(qc, y[:,-1])
        d = x[:,[1,0,2],:].copy()
        assert np.array_equal(d[:,2],x[:,2])
        assert np.all(np.any(d[:,0] != x[:,0],axis=1))
        target_d = (y[:,:2].sum(1)-y[:,-1]) if name == 'icl' else y[:,-1].copy()
        seq_hashes = [hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest() for a,b in zip(x,y)]
        assert len(set(seq_hashes)) == N
        data[name] = (x,y,d,ids,target_d)
        data_audit[name] = dict(count=N, unique_sequences=len(set(seq_hashes)),
             query_classes=int(len(np.unique(qc))), identity_verified=True,
             query_in_context_count=same.sum(1).tolist()[:8],
             all_rows_satisfy_sampler_conditions=True, donor_target_changed=bool(name=='icl'),
             receiver_sha256=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest(),
             donor_sha256=hashlib.sha256(d.tobytes()+y.tobytes()).hexdigest())
(out/'data_audit.json').write_text(json.dumps(data_audit,indent=2)+'\n')
curves = {int(float(r['training_sequences'])):r for r in
          csv.DictReader((work/'reference_learning_curves.tsv').open(),delimiter='\t')}
columns = dict(icl='icl_in_context_acc',iwl_copy_avail='ciwl_in_context_acc')
manifest = json.loads((work/'code_manifest.json').read_text())
for n,h in manifest.items(): assert sha(work/n)==h, ('code',n)
prov = dict(stages=STAGES,evaluators=EVALUATORS,count=N,batch_size=BS,conditions=len(conditions),
    checkpoint_sha256={str(s):sha(checkpoint(s)) for s in STAGES},
    evaluation_data_sha256=sha(ev),configuration_sha256=sha(cfg),feature_sha256=sha(FEATURE),
    source_sha256={n:sha(source/n) for n in expected['source_sha256']},code_sha256=manifest,
    jax=jax.__version__,equinox=eqx.__version__,numpy=np.__version__,platform=platform.platform(),
    devices=[str(d) for d in jax.devices()],no_training=True,eval_key=[0,0],
    precision='float32 dtype; float32 matrix precision; original reference default/None',
    donor='swap examples[0],examples[1]; keep label tokens and query; per-row same-checkpoint',
    orientation='margin = alternative_label logit - original_receiver_target logit',
    uncertainty='query-class cluster robust normal 95% CI; exploratory per-head CIs unadjusted')
(out/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
print('INPUTS_VERIFIED',prov['devices'],data_audit,flush=True)

def dataset_acc(m,name,donor):
    return 1-m['acc'] if donor and name=='icl' else m['acc']

def aggregate(v,name,donor=False):
    a=dataset_acc(v,name,donor)
    return dict(original_target_accuracy=float(v['acc'].mean(dtype=np.float64)),
        input_correct_accuracy=float(a.mean(dtype=np.float64)),
        alternative_choice=float(v['donor_choice'].mean(dtype=np.float64)),
        mean_margin=float(v['margin'].mean(dtype=np.float64)),
        original_target_probability=float(v['context_prob'].mean(dtype=np.float64)),
        original_target_loss=float(v['context_loss'].mean(dtype=np.float64)),
        full_original_target_loss=float(v['full_loss'].mean(dtype=np.float64)))

def channel_audit_one(model,x,y,b,d,qm,km,vm,sender,edge):
    r=patched_forward(model,x,y,b,d,qm,km,vm,sender,edge)
    a=r['transformer_output']['block_outputs'][1]['attn_output']
    errors={}
    for n,mask in [('q',qm),('k',km),('v',vm)]:
        errors[n+'_other']=jnp.max(jnp.where(~mask,jnp.abs(a[n]-b[n]),0))
        if n!='k':
            errors[n+'_target']=jnp.max(jnp.where(mask,jnp.abs(a[n]-d[n]),0))
    errors['upstream']=jnp.max(jnp.abs(r['transformer_output']['block_outputs'][0]['out']-b['l1_out']))
    errors['finite']=jnp.all(jnp.isfinite(r['out'])).astype(jnp.float32)
    return errors

channel_audit_fn=eqx.filter_jit(jax.vmap(channel_audit_one,
                        in_axes=(None,0,0,0,0,None,None,None,None,None)))

# Before formal scan, verify all six dataset/stage cells on first128 rows.
preflight, baselines = {}, {}
for stage in STAGES:
    model = restore(stage)
    preflight[str(stage)], baselines[str(stage)] = {}, {}
    for name in EVALUATORS:
        xr,y,xd,ids,target_d = data[name]
        rm,rb=batched(native_fn,model,xr[:BS],y[:BS])
        dm,db=batched(native_fn,model,xd[:BS],y[:BS])
        rb,db=jax.tree_util.tree_map(jnp.asarray,(rb,db))
        selfc=next(c for c in conditions if c['name']=='K_self')
        cr=batched(patch_fn,model,xr[:BS],y[:BS],rb,rb,selfc)
        cd=batched(patch_fn,model,xd[:BS],y[:BS],db,db,selfc)
        numeric=dict(R_noop_margin_max=float(np.max(np.abs(cr['margin']-rm['margin']))),
                     D_noop_margin_max=float(np.max(np.abs(cd['margin']-dm['margin']))))
        assert max(numeric.values())<=1e-4,numeric
        audits={}
        for audit_name in ['K_hall_p13','K_self','L1_all_to_K_all']:
            c=next(c for c in conditions if c['name']==audit_name)
            donor=rb if c.get('self_patch') else db
            vals=audit_fn(model,xr[:BS],y[:BS],rb,donor,*masks(c),jnp.asarray(c['kind']=='edge'))
            a={k:float(np.max(np.asarray(v))) for k,v in vals.items()}
            for k,v in a.items():
                if k=='final_logit_max' and audit_name!='K_self': continue
                tol=1e-4 if k in ['l1_split_max','k_target_max','final_logit_max','score_recompute_max'] else 2e-6
                assert v<=tol,(stage,name,audit_name,k,v)
            audits[audit_name]=a
        kp=batched(patch_fn,model,xr[:BS],y[:BS],rb,db,next(c for c in conditions if c['name']=='K_hall_p13'))
        ep=batched(patch_fn,model,xr[:BS],y[:BS],rb,db,next(c for c in conditions if c['name']=='L1_all_to_K_all'))
        edge_diff=float(np.max(np.abs(kp['margin']-ep['margin'])))
        assert edge_diff<=2e-4,edge_diff
        # Every mask agrees with declared head/position/channel and passes finite output.
        channel_checks={}
        for c in conditions:
            qm,km,vm,_=masks(c)
            heads=8 if c['head']==-1 else 1
            counts=[int(np.asarray(m).sum()) for m in [qm,km,vm]]
            required=[heads*8 if c['kind']=='Q' else 0,
                heads*len(c['positions'])*8 if c['kind'] in ['K','KV','edge'] else 0,
                heads*len(c['positions'])*8 if c['kind'] in ['V','KV'] else 0]
            assert counts==required,(c['name'],counts,required)
            rev=c['reverse']
            base=db if rev else rb
            donor=rb if rev or c.get('self_patch') else db
            v=channel_audit_fn(model,(xd if rev else xr)[:BS],y[:BS],base,donor,
                               *masks(c),jnp.asarray(c['kind']=='edge'))
            errors={k:float(np.max(np.asarray(a))) for k,a in v.items() if k!='finite'}
            assert np.all(np.asarray(v['finite'])==1),c['name']
            assert max(errors.values())<=2e-6,(stage,name,c['name'],errors)
            channel_checks[c['name']]=errors
        with jax.default_matmul_precision(None):
            ref=author_fn(model,jnp.asarray(xr),jnp.asarray(y),key=KEY)
        observed=float(np.asarray(ref['in_context_acc']).mean(dtype=np.float64))
        reference=float(curves[stage][columns[name]])
        assert abs(observed-reference)<=1e-9,(stage,name,observed,reference)
        baselines[str(stage)][name]=dict(observed=observed,historical=reference,exact_match=True)
        preflight[str(stage)][name]=dict(numeric=numeric,intervention_audits=audits,
            all_L1_to_K_equivalence_max=edge_diff,all_132_masks_checked=True,
            all_132_actual_channel_checks=channel_checks,
            R_first128=aggregate(cr,name),D_first128=aggregate(cd,name,True))
        (out/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
        (out/'baseline_audit.json').write_text(json.dumps(baselines,indent=2)+'\n')
        print('PREFLIGHT_CELL_PASSED',stage,name,'author_acc',observed,flush=True)
preflight_seconds=perf_counter()-t0
(out/'preflight_complete.json').write_text(json.dumps(dict(complete=True,seconds=preflight_seconds),indent=2)+'\n')
print('ALL_PREFLIGHT_PASSED',round(preflight_seconds,2),flush=True)
if args.preflight_only: sys.exit(0)

formal_start=perf_counter()
results,native_R,native_D,stage_differences,interactions={},{},{},{},{}
hp=out/'per_sequence.h5'

def store_metrics(f,group,values):
    if group in f: del f[group]
    g=f.create_group(group)
    for n,v in values.items():
        v=np.asarray(v)
        assert len(v)==N and np.isfinite(v).all(),(group,n)
        g.create_dataset(n,data=v,compression='gzip')
    g.attrs['complete']=True
    f.flush()

def read_metrics(f,group): return {n:np.asarray(v) for n,v in f[group].items()}

def summarize(stage,name,c,v,base,ids,y):
    qc=ids[:,-1,0]
    delta=v['margin']-base['margin']
    # Reverse ICL keeps common margin orientation but actual D correctness is the opposite.
    acc=dataset_acc(v,name,c['reverse'])
    bacc=dataset_acc(base,name,c['reverse'])
    cs=cluster_stats(delta,qc)
    acs=cluster_stats(acc-bacc,qc)
    ps=cluster_stats(v['context_prob']-base['context_prob'],qc)
    ls=cluster_stats(v['context_loss']-base['context_loss'],qc)
    return dict(stage=stage,evaluator=name,condition=c['name'],kind=c['kind'],head=c['head'],
        sender=c.get('sender',-2),reverse=c['reverse'],count=N,
        **aggregate(v,name,c['reverse']),
        delta_margin=cs['mean'],delta_margin_ci95_low=cs['ci95_low'],delta_margin_ci95_high=cs['ci95_high'],
        query_classes=cs['query_classes'],delta_input_correct_accuracy=acs['mean'],
        delta_accuracy_ci95_low=acs['ci95_low'],delta_accuracy_ci95_high=acs['ci95_high'],
        delta_original_target_probability=ps['mean'],delta_original_target_loss=ls['mean'],
        delta_full_original_target_loss=float((v['full_loss']-base['full_loss']).mean(dtype=np.float64)),
        mean_delta_attention_by_head=(v['attn_margin']-base['attn_margin']).mean(axis=0,dtype=np.float64).tolist(),
        mean_delta_score_by_head=(v['score_margin']-base['score_margin']).mean(axis=0,dtype=np.float64).tolist())

with h5py.File(hp,'a') as f:
    if 'provenance' in f.attrs: assert json.loads(f.attrs['provenance'])==prov
    else: f.attrs['provenance']=json.dumps(prov,sort_keys=True)
    for name,(xr,y,xd,ids,target_d) in data.items():
        if f'inputs/{name}' not in f:
            g=f.create_group(f'inputs/{name}')
            g.create_dataset('receiver_examples',data=xr,compression='gzip')
            g.create_dataset('input_labels',data=y)
            g.create_dataset('receiver_target',data=y[:,-1])
            g.create_dataset('donor_target',data=target_d)
            g.create_dataset('class_exemplar_ids',data=ids)
            g.create_dataset('row_id',data=np.arange(N))
            g.attrs['donor_example_permutation']=[1,0,2]
    for stage in STAGES:
        model=restore(stage)
        results[str(stage)],native_R[str(stage)],native_D[str(stage)]={},{},{}
        for name in EVALUATORS:
            xr,y,xd,ids,target_d=data[name]
            prefix=f'{stage}/{name}'
            rm,rb=batched(native_fn,model,xr,y)
            dm,db=batched(native_fn,model,xd,y)
            rb,db=jax.tree_util.tree_map(jnp.asarray,(rb,db))
            selfc=next(c for c in conditions if c['name']=='K_self')
            cr=batched(patch_fn,model,xr,y,rb,rb,selfc)
            cd=batched(patch_fn,model,xd,y,db,db,selfc)
            for control,raw in [(cr,rm),(cd,dm)]:
                assert np.max(np.abs(control['margin']-raw['margin']))<=1e-4
                assert abs(float((control['acc']-raw['acc']).mean()))<=5/N+1e-9
            wrapper=float(cr['acc'].mean(dtype=np.float64))
            assert abs(wrapper-baselines[str(stage)][name]['observed'])<=5/N+1e-9
            baselines[str(stage)][name]['strict_wrapper_acc']=wrapper
            (out/'baseline_audit.json').write_text(json.dumps(baselines,indent=2)+'\n')
            store_metrics(f,prefix+'/native_R',cr);store_metrics(f,prefix+'/native_D',cd)
            native_R[str(stage)][name]=aggregate(cr,name)
            native_D[str(stage)][name]=aggregate(cd,name,True)
            results[str(stage)][name]={}
            print('FORMAL_CELL_START',stage,name,'R',wrapper,'D_correct',native_D[str(stage)][name]['input_correct_accuracy'],flush=True)
            for i,c in enumerate(conditions):
                group=prefix+'/patch/'+c['name']
                if group in f and f[group].attrs.get('complete',False): v=read_metrics(f,group)
                else:
                    ex=xd if c['reverse'] else xr
                    base=db if c['reverse'] else rb
                    donor=rb if c['reverse'] or c.get('self_patch') else db
                    v=batched(patch_fn,model,ex,y,base,donor,c)
                    store_metrics(f,group,v)
                    f[group].attrs['condition']=json.dumps(c,sort_keys=True)
                s=summarize(stage,name,c,v,cd if c['reverse'] else cr,ids,y)
                results[str(stage)][name][c['name']]=s
                if c['name']=='K_self':
                    assert np.max(np.abs(v['margin']-cr['margin']))<=1e-4
                    assert np.array_equal(v['acc'],cr['acc'])
                if c['name']=='L1_all_to_K_all':
                    assert np.max(np.abs(v['margin']-f[prefix+'/patch/K_hall_p13/margin'][:]))<=2e-4
                (out/'progress.json').write_text(json.dumps(dict(stage=stage,evaluator=name,condition=c['name'],
                    completed_in_cell=i+1,conditions_per_cell=len(conditions),elapsed_seconds=perf_counter()-formal_start),indent=2)+'\n')
                print('CONDITION',stage,name,i+1,c['name'],'delta_margin',round(s['delta_margin'],5),
                    'correct',round(s['input_correct_accuracy'],4),'seconds',round(perf_counter()-formal_start,1),flush=True)
            del rb,db
    for name in EVALUATORS:
        qc=data[name][3][:,-1,0]
        stage_differences[name]={};interactions[name]={}
        for c in ['K_hall_p13','V_hall','Q_hall','KV_hall','L1_0_to_K_hall']:
            ds={s:f[f'{s}/{name}/patch/{c}/margin'][:]-f[f'{s}/{name}/native_R/margin'][:] for s in STAGES}
            stage_differences[name][c]={f'{late}_minus_{early}':cluster_stats(ds[late]-ds[early],qc)
                for early,late in [(4400000,8000000),(4400000,20000000)]}
        for stage in STAGES:
            p=f'{stage}/{name}'
            effect=f[p+'/patch/KV_hall/margin'][:]-f[p+'/patch/K_hall_p13/margin'][:]-f[p+'/patch/V_hall/margin'][:]+f[p+'/native_R/margin'][:]
            interactions[name][str(stage)]=cluster_stats(effect,qc)
    f.attrs['complete']=True
    f.flush()
rows=[r for st in results.values() for evs in st.values() for r in evs.values()]
assert len(rows)==792
flat=[{k:v for k,v in r.items() if not isinstance(v,list)} for r in rows]
with (out/'summary.tsv').open('w') as fp:
    w=csv.DictWriter(fp,fieldnames=list(flat[0]),delimiter='\t');w.writeheader();w.writerows(flat)
summary=dict(complete=True,native_R=native_R,native_D=native_D,results=results,
    stage_differences=stage_differences,KV_logit_interaction=interactions,
    preflight_seconds=preflight_seconds,formal_seconds=perf_counter()-formal_start,total_seconds=perf_counter()-t0)
for stage in STAGES: assert sha(checkpoint(stage))==expected['checkpoint_sha256_by_stage'][str(stage)]
assert sha(ev)==expected['evaluation_data_sha256'] and sha(cfg)==expected['configuration_sha256']
assert sha(FEATURE)==expected['feature_sha256']
for n,h in expected['source_sha256'].items(): assert sha(source/n)==h
for n,h in manifest.items(): assert sha(work/n)==h
(out/'post_input_audit.json').write_text(json.dumps(dict(unchanged=True),indent=2)+'\n')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('COMPLETED',len(rows),'condition_results','seconds',round(perf_counter()-t0,2),flush=True)
