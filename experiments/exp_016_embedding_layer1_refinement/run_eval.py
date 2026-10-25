"""Stage 2 only: split embeddings and L1; fixed late L2/readout; author forward."""
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import itertools
import json
import platform
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
args = ap.parse_args()
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

START = perf_counter()
OUT = work / 'outputs'
OUT.mkdir(exist_ok=True)
N, BS = 5000, 128
STAGES = {'E': 4400000, 'L': 20000000}
SETS = ['icl']
PARTS = ['example_embed', 'label_embed', 'pos_embed', 'layer1']
ANCHORS = {'EEEE': 'EELL', 'EEEL': 'ELLL', 'LLLE': 'LELL', 'LLLL': 'LLLL'}
parent = root / 'exp_015_layer_strategy_reversal'
parent_expected = json.loads((work / 'parent_expected.json').read_text())
CODES = [''.join(v) for v in itertools.product('EL', repeat=4)]
KEY = jax.random.PRNGKey(0)
expected = json.loads((work / 'expected_inputs.json').read_text())


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def checkpoint(stage):
    return root / f'exp_002_main_reproduction/runs/main_resume_00002950016/checkpoints/{stage:011d}.eqx'


cfg = root / 'exp_002_main_reproduction/runs/main/config.json'
ev = root / 'exp_002_main_reproduction/runs/default_eval/eval_data.h5'
feature = root / 'exp_002_main_reproduction/features/omniglot_features_norotate.h5'
inputs = {str(cfg): expected['configuration_sha256'],
          str(ev): expected['evaluation_data_sha256'],
          str(feature): expected['feature_sha256']}
inputs.update({str(source / n): h for n, h in expected['source_sha256'].items()})
inputs.update({str(checkpoint(t)): expected['checkpoint_sha256_by_stage'][str(t)]
               for t in STAGES.values()})
inputs[str(work / 'reference_learning_curves.tsv')] = expected['baseline_curve_sha256']
inputs[str(work / 'author_forward_metrics.py')] = expected['author_forward_sha256']
inputs.update({str(parent / name): digest for name, digest in parent_expected['files'].items()})
manifest = json.loads((work / 'code_manifest.json').read_text())
for name, digest in manifest.items():
    assert sha(work / name) == digest, ('code identity', name)
for name, digest in inputs.items():
    assert sha(name) == digest, ('input identity', name)
assert jax.default_backend() == 'gpu', jax.devices()
assert jax.config.jax_default_matmul_precision is None
assert json.loads((work / 'conditions.json').read_text()) == [
    {'id': c, 'sources': dict(zip(PARTS, c))} for c in CODES]
opts = main_utils.get_opts_from_json_file(cfg)
assert (opts.depth, opts.num_heads, opts.d_model) == (2, 8, 64)
assert opts.mlp_ratio is None and opts.pos_embedding_type == 'ape'
model0 = main_utils.get_model_from_opts(opts, (512,))
assert isinstance(model0.transformer.unembed, eqx.nn.Identity)
assert all(isinstance(b.norm1, eqx.nn.LayerNorm) for b in model0.transformer.blocks)
for obj in jax.tree_util.tree_leaves(model0, is_leaf=lambda x: isinstance(x, eqx.nn.Dropout)):
    if isinstance(obj, eqx.nn.Dropout):
        assert obj.p == 0
optimizer = main_utils.get_optimizer_from_opts(opts)
template = dict(iter=-1, model=model0,
                opt_state=optimizer.init(eqx.filter(model0, eqx.is_array)),
                seeds=dict(eval_model_seed=KEY, train_data_seed=KEY, train_model_seed=KEY))


def arrays(tree):
    return [x for x in jax.tree_util.tree_leaves(tree) if eqx.is_array(x)]


def digest_arrays(tree):
    h = hashlib.sha256()
    for x in arrays(tree):
        a = np.asarray(x)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


models = {}
for key, stage in STAGES.items():
    ck = eqx.tree_deserialise_leaves(checkpoint(stage), template)
    assert int(ck['iter']) == stage and int(ck['opt_state'][0].count) == stage // 32
    assert all(np.isfinite(np.asarray(x)).all() for x in arrays(ck))
    models[key] = ck['model']
assert jax.tree_util.tree_structure(models['E']) == jax.tree_util.tree_structure(models['L'])
assert eqx.tree_equal(eqx.filter(models['E'], lambda x: not eqx.is_array(x)),
                      eqx.filter(models['L'], lambda x: not eqx.is_array(x))) is True
native_digests = {k: digest_arrays(m) for k, m in models.items()}


SELECTORS = [lambda m: m.example_embed, lambda m: m.label_embed,
             lambda m: m.pos_embed, lambda m: m.transformer.blocks[0]]
FIXED_SELECTORS = [lambda m: m.transformer.blocks[1], lambda m: (m.transformer.norm, m.unembed)]


def make_model(code):
    m = models['L']
    for select, origin in zip(SELECTORS, code):
        m = eqx.tree_at(select, m, select(models[origin]))
    return m


partition_audit = {}
for origin, m in models.items():
    full = arrays(m)
    grouped = [x for select in SELECTORS + FIXED_SELECTORS for x in arrays(select(m))]
    assert len(grouped) == len(full)
    assert len(set(map(id, grouped))) == len(grouped)
    assert set(map(id, full)) == set(map(id, grouped))
    partition_audit[origin] = dict(total_arrays=len(full),
        arrays_per_part={p: len(arrays(s(m))) for p, s in zip(PARTS + ['fixed_L2', 'fixed_readout'], SELECTORS + FIXED_SELECTORS)})
variant_audit = {}
for code in CODES:
    m = make_model(code)
    for select, origin in zip(SELECTORS, code):
        a, b = arrays(select(m)), arrays(select(models[origin]))
        assert len(a) == len(b)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    for select in FIXED_SELECTORS:
        for x, y in zip(arrays(select(m)), arrays(select(models['L']))):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    variant_audit[code] = dict(parameter_sha256=digest_arrays(m),
                              all_source_arrays_exact=True, fixed_downstream_exact=True)
    if code in ANCHORS:
        assert variant_audit[code]['parameter_sha256'] == parent_expected['parameter_digests'][ANCHORS[code]]

with h5py.File(feature, 'r') as f:
    features = np.asarray(f['resnet18/224/feat'][:1600, :5], np.float32)
    assert np.array_equal(f['original_class'][:1600], np.arange(1600))
    assert np.all(f['augmentation'][:1600] == 0)
lookup = {}
for cls in range(1600):
    for sample in range(5):
        k = hashlib.sha256(features[cls, sample].tobytes()).digest()
        assert k not in lookup
        lookup[k] = (cls, sample)
data, identities, data_audit = {}, {}, {}
with h5py.File(ev, 'r') as f:
    for name in SETS:
        x = np.asarray(f[name]['examples'], np.float32)
        y = np.asarray(f[name]['labels'], np.int32)
        assert x.shape == (N, 3, 512) and y.shape == (N, 3)
        assert np.isfinite(x).all() and np.all(y[:, 0] != y[:, 1])
        assert np.all((y[:, :2] == y[:, -1, None]).sum(1) == 1)
        ids = np.asarray([lookup[hashlib.sha256(v.tobytes()).digest()]
                          for v in x.reshape(-1, 512)], np.int32).reshape(N, 3, 2)
        qc = ids[:, -1, 0]
        same = ids[:, :2, 0] == qc[:, None]
        if name == 'iwl_copy_avail':
            assert np.all(same.sum(1) == 0)
            assert np.array_equal(qc, y[:, -1])
        else:
            assert np.all(same.sum(1) == 1)
            assert np.array_equal(y[:, :2][same], y[:, -1])
            if name == 'icl':
                assert np.all(np.sort(y[:, :2], axis=1) == np.array([0, 1]))
            elif name == 'train_eval':
                assert np.array_equal(y[:, -1], qc)
            elif name == 'flip_icl':
                assert np.all(y[:, -1] != qc)
                assert np.all((y[:, :2] == qc[:, None]).sum(1) == 1)
                assert np.array_equal(y[:, :2].sum(1) - y[:, -1], qc)
        data[name], identities[name] = (x, y), ids
        data_audit[name] = dict(count=N, query_classes=int(len(np.unique(qc))),
            identity_verified=True, input_sha256=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest(),
            target_label_position_counts=np.bincount(np.argmax(y[:, :2] == y[:, -1, None], axis=1), minlength=2).tolist())
write_json(OUT / 'data_audit.json', data_audit)

curves = {int(float(r['training_sequences'])): r for r in
          csv.DictReader((work / 'reference_learning_curves.tsv').open(), delimiter='\t')}
columns = dict(train_eval='train_in_context_acc', icl='icl_in_context_acc',
               iwl_copy_avail='ciwl_in_context_acc', flip_icl='flip_in_context_acc')
p = argparse.ArgumentParser()
opto.add_args_to_parser(p)
author_opts = p.parse_args([])
author_fn = main.make_batched_fn(make_forward_fn_ic_only(author_opts), 1000)
baseline_audit = {}
for origin, model in models.items():
    baseline_audit[origin] = {}
    for name, (x, y) in data.items():
        v = author_fn(model, jnp.asarray(x), jnp.asarray(y), key=KEY)
        count = int(np.asarray(v['in_context_acc']).sum())
        reference = float(curves[STAGES[origin]][columns[name]])
        assert count == round(reference * N), ('author baseline mismatch', origin, name, count, reference)
        baseline_audit[origin][name] = dict(default_correct_count=count, historical_accuracy=reference)
        print('baseline', origin, name, count, 'elapsed', round(perf_counter()-START, 2), flush=True)

jax.config.update('jax_default_matmul_precision', 'float32')


def metrics(result, y):
    z = result['out'][-1]
    target = y[-1]
    other = y[0] + y[1] - target
    margin = z[target] - z[other]
    pred = jnp.where(z[y[0]] > z[y[1]], y[0],
                    jnp.where(z[y[1]] > z[y[0]], y[1], jnp.minimum(y[0], y[1])))
    full_pred = jnp.argmax(z)
    correct_slot = jnp.argmax(y[:2] == target)
    incorrect_slot = 1 - correct_slot
    blocks = result['transformer_output']['block_outputs']
    a1 = blocks[0]['attn_output']['attn_scores'][0]
    a2 = blocks[1]['attn_output']['attn_scores'][0, :, 4]
    s2 = blocks[1]['attn_output']['attn_pre_softmax'][0, :, 4]
    cp, ip = 2*correct_slot+1, 2*incorrect_slot+1
    return dict(pair_margin=margin, pair_acc=(pred == target).astype(jnp.float32),
                pair_prob=jax.nn.sigmoid(margin), pair_loss=jax.nn.softplus(-margin),
                pair_pred=pred, full_pred=full_pred,
                full_acc=(full_pred == target).astype(jnp.float32),
                full_other_label=(full_pred == other).astype(jnp.float32),
                full_loss=jax.scipy.special.logsumexp(z)-z[target],
                outside_pair=((full_pred != y[0]) & (full_pred != y[1])).astype(jnp.float32),
                l1_previous=(a1[:, 1, 0]+a1[:, 3, 2])/2,
                l1_self=(a1[:, 1, 1]+a1[:, 3, 3])/2,
                l2_attention_delta=a2[:, cp]-a2[:, ip],
                l2_score_delta=s2[:, cp]-s2[:, ip])


def one(model, x, y):
    return metrics(model.call_with_all_aux(examples=x, labels=y, key=KEY), y)


native_fn = eqx.filter_jit(jax.vmap(one, in_axes=(None, 0, 0)))
graft_opts = p.parse_args([])
graft_opts.opto_graft_out_model_ckpt = str(checkpoint(STAGES['L']))
graft_opts.opto_graft_out_model_cfg = str(cfg)
graft_opts.opto_graft_out_model_from_layer = 0
graft_call = opto.make_fn_from_opts(graft_opts)


def graft_one(model, x, y):
    return metrics(graft_call(model=model, x=x, y=y, key=KEY), y)


graft_fn = eqx.filter_jit(jax.vmap(graft_one, in_axes=(None, 0, 0)))
numeric_audit = {}
for name, (x, y) in data.items():
    xx, yy = jnp.asarray(x[:BS]), jnp.asarray(y[:BS])
    for code in ['LLLL', 'EEEE']:
        got = {k: np.asarray(v) for k, v in native_fn(make_model(code), xx, yy).items()}
        ref_fn = graft_fn if code == 'EEEE' else native_fn
        ref_model = models['E'] if code == 'EEEE' else models['L']
        ref = {k: np.asarray(v) for k, v in ref_fn(ref_model, xx, yy).items()}
        err = float(np.max(np.abs(got['pair_margin']-ref['pair_margin'])))
        assert err <= 1e-4, (name, code, err)
        assert np.array_equal(got['pair_pred'], ref['pair_pred'])
        assert np.array_equal(got['full_pred'], ref['full_pred'])
        numeric_audit[f'{code}/{name}'] = dict(margin_max_error=err, pair_and_full_predictions_equal=True)
with h5py.File(parent / 'outputs/per_sequence.h5', 'r') as previous:
    pp = json.loads(previous.attrs['provenance'])
    assert pp['code_manifest_sha256'] == parent_expected['files']['code_manifest.json']
    assert pp['precision'] == 'float32 dtype / float32 matmul' and pp['batch_size'] == BS and pp['key'] == [0, 0]
    for name, (x, y) in data.items():
        np.testing.assert_array_equal(np.asarray(previous['inputs/'+name+'/labels']), y)
        np.testing.assert_array_equal(np.asarray(previous['inputs/'+name+'/identities']), identities[name])
        for code, old_code in ANCHORS.items():
            got = {k: np.asarray(v) for k, v in native_fn(make_model(code), jnp.asarray(x[:BS]), jnp.asarray(y[:BS])).items()}
            assert previous[old_code+'/'+name].attrs['completed']
            errors = {}
            for key, a in got.items():
                b = np.asarray(previous[old_code+'/'+name+'/'+key][:BS])
                assert a.shape == b.shape
                err = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
                assert err <= 1e-4, (code, name, key, err)
                if key in ['pair_pred','full_pred','pair_acc','full_acc','full_other_label','outside_pair']:
                    np.testing.assert_array_equal(a, b)
                errors[key] = err
            numeric_audit['parent/'+code+'/'+name] = dict(parent_code=old_code, max_error_by_metric=errors)
write_json(OUT / 'preflight.json', dict(passed=True, partition=partition_audit,
           variants=variant_audit, numerics=numeric_audit,
           preflight_elapsed_seconds=perf_counter()-START))

provenance = dict(stages=STAGES, parts=PARTS, codes=CODES, input_sha256=inputs,
    code_sha256=manifest, code_manifest_sha256=sha(work/'code_manifest.json'),
    no_training=True, fixed_global_parameter_interventions=True,
    fixed_downstream='20M layer2/final norm/unembed', anchors=ANCHORS,
    parent_expected=parent_expected,
    input_donor_selection=False, batch_size=BS, sequences_per_evaluator=N,
    evaluators=SETS, precision='float32 dtype / float32 matmul', key=[0, 0],
    jax=jax.__version__, equinox=eqx.__version__, python=platform.python_version(),
    devices=[str(v) for v in jax.devices()], native_parameter_digests=native_digests)
hp = OUT / 'per_sequence.h5'
rows = []
with h5py.File(hp, 'a') as f:
    if 'provenance' in f.attrs:
        assert json.loads(f.attrs['provenance']) == provenance, 'Resume provenance mismatch'
    else:
        f.attrs['provenance'] = json.dumps(provenance, sort_keys=True)
        for name, (_, y) in data.items():
            g = f.create_group('inputs/'+name)
            g.create_dataset('labels', data=y)
            g.create_dataset('identities', data=identities[name])
    write_json(OUT / 'provenance.json', provenance)
    for code in ['EEEE', 'LLLL'] + [c for c in CODES if c not in ['EEEE', 'LLLL']]:
        model = make_model(code)
        for name, (x, y) in data.items():
            group = code+'/'+name
            if group in f and f[group].attrs.get('completed', False):
                saved = json.loads(f[group].attrs['summary'])
                assert (saved['code'], saved['evaluator'], saved['count']) == (code, name, N)
                for key, stored in f[group].items():
                    a = np.asarray(stored)
                    assert a.shape[0] == N and np.isfinite(a).all()
                    if key in saved and a.ndim == 1:
                        assert abs(a.mean(dtype=np.float64)-saved[key]) < 1e-9
                assert int(np.asarray(f[group+'/pair_acc']).sum()) == saved['correct_count']
                if code == 'LLLL':
                    ref = baseline_audit[code[0]][name]['default_correct_count']
                    assert abs(saved['correct_count']-ref) <= 5
                    baseline_audit[code[0]][name].update(strict_correct_count=saved['correct_count'],
                        strict_minus_default_count=saved['correct_count']-ref)
                rows.append(saved)
                continue
            assert perf_counter()-START < 9000, '2.5 hour safe-boundary budget reached'
            pieces = []
            for start in range(0, N, BS):
                v = native_fn(model, jnp.asarray(x[start:start+BS]), jnp.asarray(y[start:start+BS]))
                pieces.append({k: np.asarray(a) for k, a in v.items()})
            values = {k: np.concatenate([a[k] for a in pieces]) for k in pieces[0]}
            assert all(a.shape[0] == N and np.isfinite(a).all() for a in values.values())
            summary = {k: float(a.mean(dtype=np.float64)) for k, a in values.items() if a.ndim == 1 and k not in ['pair_pred', 'full_pred']}
            summary.update(code=code, evaluator=name, count=N,
                           correct_count=int(values['pair_acc'].sum()),
                           full_correct_count=int(values['full_acc'].sum()))
            if code == 'LLLL':
                ref = baseline_audit[code[0]][name]['default_correct_count']
                assert abs(summary['correct_count']-ref) <= 5, ('strict baseline mismatch', code, name)
                baseline_audit[code[0]][name].update(strict_correct_count=summary['correct_count'],
                                                      strict_minus_default_count=summary['correct_count']-ref)
            if group in f:
                del f[group]
            g = f.create_group(group)
            for key, a in values.items():
                g.create_dataset(key, data=a, compression='gzip')
            g.attrs['summary'] = json.dumps(summary, sort_keys=True)
            g.attrs['completed'] = True
            f.flush()
            rows.append(summary)
            write_json(OUT / 'progress.json', dict(completed=len(rows), total=64,
                last=group, elapsed_seconds=perf_counter()-START, last_summary=summary))
            print('completed', group, 'pair_acc', summary['pair_acc'], 'full_acc', summary['full_acc'],
                  'elapsed', round(perf_counter()-START, 2), flush=True)
    assert len(rows) == 64
    for code in CODES:
        for name in SETS:
            assert f[code+'/'+name].attrs['completed']
            assert f[code+'/'+name+'/pair_margin'].shape == (N,)
bridge_audit = {}
with h5py.File(parent / 'outputs/per_sequence.h5', 'r') as previous, h5py.File(hp, 'r') as current:
    for code, old_code in ANCHORS.items():
        for name in SETS:
            errors = {}
            for key in current[code+'/'+name]:
                a = np.asarray(current[code+'/'+name+'/'+key])
                b = np.asarray(previous[old_code+'/'+name+'/'+key])
                assert a.shape == b.shape
                err = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
                assert err <= 1e-4, (code, name, key, err)
                if key in ['pair_pred','full_pred','pair_acc','full_acc','full_other_label','outside_pair']:
                    np.testing.assert_array_equal(a, b)
                errors[key] = err
            bridge_audit[code+'/'+name] = dict(parent_code=old_code, count=N, max_error_by_metric=errors)
write_json(OUT / 'parent_bridge_audit.json', dict(passed=True, groups=16, results=bridge_audit))
for k, m in models.items():
    assert digest_arrays(m) == native_digests[k], 'Source model mutated'
for path, digest in inputs.items():
    assert sha(path) == digest, ('Input changed', path)
write_json(OUT / 'baseline_audit.json', baseline_audit)
write_json(OUT / 'post_input_audit.json', dict(passed=True, sha256=inputs,
           source_models_unchanged=True))
with (OUT / 'summary.tsv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
    w.writeheader()
    w.writerows(rows)
write_json(OUT / 'summary.json', rows)
write_json(OUT / 'evaluation_finished.json', dict(status='completed', conditions=16,
    evaluator_conditions=64, formal_sequences=320000, new_condition_sequences=240000,
    parent_anchor_recheck_sequences=80000, default_baseline_sequences=40000,
    elapsed_seconds=perf_counter()-START, per_sequence_sha256=sha(hp),
    completed_utc=datetime.now(timezone.utc).isoformat()))
print('EVALUATION_COMPLETED', round(perf_counter()-START, 2), flush=True)
