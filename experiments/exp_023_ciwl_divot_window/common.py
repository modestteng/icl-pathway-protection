"""Frozen author model, approved inputs, same-checkpoint cache interventions."""
from pathlib import Path
import hashlib
import json
import sys
import time

W = Path(__file__).resolve().parent
O = W / 'outputs'
P = W.parent / 'exp_024_l2_label_k_path'
PO = P / 'outputs'
ROOT = W.parent
SOURCE = ROOT / 'reproduction_support/source'
sys.path.insert(0, str(SOURCE))

import numpy as np
import h5py
import jax
import jax.numpy as jnp
import equinox as eqx
import main_utils
import main
import opto

# Evaluation arithmetic adapter; weights, dtype and scientific conditions are unchanged.
# The original default-precision preflight and all original input identities are preserved.
jax.config.update('jax_default_matmul_precision', 'float32')

from make_fresh import make_confirmation, _validate, _feature_lookup

SETTINGS = json.loads((W / 'settings.json').read_text())
KEY = jax.random.PRNGKey(0)
N = SETTINGS['n']
BS = SETTINGS['batch_size']
CFG = ROOT / 'exp_002_main_reproduction/runs/main/config.json'
EVAL = ROOT / 'exp_002_main_reproduction/runs/default_eval/eval_data.h5'
FEATURE = ROOT / 'exp_002_main_reproduction/features/omniglot_features_norotate.h5'
EXPECTED = {
    'config': '5c6cadc87ba1f0796681116f8c222e742200be51658481c579776e5e8b585223',
    'eval': '634ee065b3183599a409c325960ad5fbe4388236a4fc60143794467da3c5efec',
    'feature': 'd1dd111c34d3bf17829e3191f0c01c6821e3932c7f3ca6ef9c43b49f7f66af70',
    'source': {
        'main.py': '85814cae9c85cadd498d7cb1376708c9f9f2bcac237127943ff4697268754815',
        'main_utils.py': '07bfe1c34366643bc89d8b9a4194c195846e462be31ddf432734b0a817b8727d',
        'models.py': '2509f2a436dd16336f496769bb3d43df89e579cefd178b394d409b46c8892f99',
        'samplers.py': 'd285d7d6796996162c0e445bd4d93cdd54c8e9d400bb69e81d054e1bc7a841da',
        'opto.py': '46cbdcc2ef3c7a8688dfb365dcdc7fac753c05e96e8fee47373e2b1a2ad5d1b3'},
    'checkpoints': {
        0: 'cc5d646d6dd6f6e0ff5632921904d57577e19ef11df59d1150ca3fb2437646c0',
        2000000: '6cc17173f9f19c5a8c565d387b5d1429f4486a889c5e5fd523686107f8524c62',
        4400000: '627972690994b53311e3ced2714afc316f4b64687706d8d37e1b7e47457da7b5',
        8000000: 'c6a1d4669004061b82a93d2bbcd37694c4a7ce77547d79cd6e2c2147d840dcac',
        20000000: '95b0609902e00b4de117ec05f21293ccfc490ff38843acac144a07033ebedd0d'}
}
O.mkdir(exist_ok=True)
PO.mkdir(parents=True, exist_ok=True)


def write_json(path, value):
    path = Path(path)
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def arrays(tree):
    return [a for a in jax.tree_util.tree_leaves(tree) if eqx.is_array(a)]


def digest_arrays(tree):
    h = hashlib.sha256()
    for x in arrays(tree):
        a = np.asarray(x)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def checkpoint_path(stage):
    it = -(-int(stage) // 32) * 32
    folder = 'main' if it < 3000000 else 'main_resume_00002950016'
    return ROOT / 'exp_002_main_reproduction/runs' / folder / 'checkpoints' / f'{it:011d}.eqx', it


def get_template():
    opts = main_utils.get_opts_from_json_file(CFG)
    assert (opts.depth, opts.num_heads, opts.d_model) == (2, 8, 64)
    assert opts.mlp_ratio is None and opts.pos_embedding_type == 'ape'
    assert opts.model_output_classes == 12800 and opts.optimizer == 'adam'
    assert opts.train_bs == 32 and opts.lr == 1e-5
    model = main_utils.get_model_from_opts(opts, (512,))
    for obj in jax.tree_util.tree_leaves(model, is_leaf=lambda x: isinstance(x, eqx.nn.Dropout)):
        if isinstance(obj, eqx.nn.Dropout):
            assert obj.p == 0
    optimizer = main_utils.get_optimizer_from_opts(opts)
    return dict(iter=-1, model=model, opt_state=optimizer.init(eqx.filter(model, eqx.is_array)),
                seeds=dict(eval_model_seed=KEY, train_data_seed=KEY, train_model_seed=KEY))


def load_checkpoint(stage, template):
    path, it = checkpoint_path(stage)
    ck = eqx.tree_deserialise_leaves(path, template)
    assert int(ck['iter']) == it and int(ck['opt_state'][0].count) == it // 32
    assert all(np.isfinite(np.asarray(a)).all() for a in arrays(ck))
    return ck['model']


def verify_inputs(stages):
    record = dict(source={}, checkpoints={})
    for name, expected in EXPECTED['source'].items():
        actual = sha(SOURCE / name)
        assert actual == expected, ('frozen source', name)
        record['source'][name] = actual
    for name, path in [('config', CFG), ('eval', EVAL), ('feature', FEATURE)]:
        actual = sha(path)
        assert actual == EXPECTED[name], ('input', name)
        record[name] = dict(path=str(path), sha256=actual)
    manifest = json.loads((W / 'code_manifest.json').read_text())
    assert all(sha(W / name) == value for name, value in manifest.items()), 'Code identity'
    record['code_manifest_sha256'] = sha(W / 'code_manifest.json')
    design = json.loads((W / 'design.json').read_text())
    assert design.get('scientific_design_frozen_before_fresh_data', design['frozen_before_fresh_data'])
    assert design['settings_sha256'] == sha(W / 'settings.json')
    assert design['protocol_sha256'] == sha(W / 'protocol.md')
    assert design['code_manifest_sha256'] == record['code_manifest_sha256']
    record['design_sha256'] = sha(W / 'design.json')
    record['time_selection_sha256'] = sha(O / 'time_selection.json')
    record['path_protocol_sha256'] = sha(P / 'protocol.md')
    for stage in stages:
        path, it = checkpoint_path(stage)
        value = sha(path)
        if stage in EXPECTED['checkpoints']:
            assert value == EXPECTED['checkpoints'][stage], ('checkpoint', stage)
        record['checkpoints'][str(stage)] = dict(path=str(path), actual_iter=it, sha256=value,
              previously_registered=stage in EXPECTED['checkpoints'])
    assert jax.default_backend() == 'gpu' and jax.__version__ == '0.4.26'
    assert eqx.__version__ == '0.11.4'
    assert jax.config.jax_default_matmul_precision == 'float32' and not jax.config.jax_enable_x64
    if (W / 'execution_repair.json').exists():
        repair = json.loads((W / 'execution_repair.json').read_text())
        assert repair['settings_sha256'] == sha(W / 'settings.json')
        snapshot = json.loads((O / 'repair_input_snapshot.json').read_text())
        for name in ['confirmation_data.h5', 'discovery_data.h5', 'time_selection.json']:
            assert sha(O / name) == snapshot[name], ('repaired input changed', name)
        record['execution_repair'] = dict(precision='float32', snapshot=snapshot,
              scientific_settings_unchanged=True, source=sha(W / 'execution_repair.json'))
    return record


def load_inputs():
    old = {}
    with h5py.File(EVAL, 'r') as f:
        for name in SETTINGS['eval_sets']:
            x, y = np.asarray(f[name]['examples'], np.float32), np.asarray(f[name]['labels'], np.int32)
            assert x.shape == (N, 3, 512) and y.shape == (N, 3)
            old[name] = (x, y)
    with h5py.File(FEATURE, 'r') as f:
        features = np.asarray(f['resnet18/224/feat'][:1600, :5], np.float32)
        assert np.array_equal(f['original_class'][:1600], np.arange(1600))
        assert np.all(f['augmentation'][:1600] == 0)
    lookup = _feature_lookup(features)
    old_ids = {}
    for name in SETTINGS['eval_sets']:
        old[name], old_ids[name], _ = _validate(name, old[name], features, lookup, expected_n=N)
    discovery, dids, daudit = make_confirmation(features, old, O / 'discovery_data.h5',
                         seed=SETTINGS['discovery_seed'], n=N)
    confirmation, cids, caudit = make_confirmation(features, old, O / 'confirmation_data.h5',
                         seed=SETTINGS['confirmation_seed'], n=N)
    write_json(O / 'fresh_input_audit.json', dict(discovery=daudit, confirmation=caudit))
    with h5py.File(O / 'original_data_identity.h5', 'a') as f:
        for name in SETTINGS['eval_sets']:
            if name not in f:
                g = f.create_group(name)
                g.create_dataset('labels', data=old[name][1])
                g.create_dataset('identities', data=old_ids[name])
    return old, discovery, confirmation


VALID = np.tri(5, dtype=bool)
ZS = np.zeros((1, 8, 5, 5), bool)
PREV = ZS.copy()
PREV[0, 0, 1, 0] = PREV[0, 0, 3, 2] = True
H6 = ZS.copy()
H6[0, 6, 1, 0] = H6[0, 6, 3, 2] = True
SELF = ZS.copy()
SELF[0, 0, 1, 1] = SELF[0, 0, 3, 3] = True
LABEL_ROWS = ZS.copy()
LABEL_ROWS[0, 0, 1, :2] = True
LABEL_ROWS[0, 0, 3, :4] = True
ZKV = np.zeros((1, 8, 5, 8), bool)
LABEL_KV = ZKV.copy()
LABEL_KV[:, :, 1, :] = LABEL_KV[:, :, 3, :] = True


def attn(r, layer):
    return r['transformer_output']['block_outputs'][layer]['attn_output']


def forward(model, x, y, score_value, score_mask, k_value, k_mask, v_value, v_mask):
    cache, mask = opto.get_default_cache_and_mask(2)
    for layer, name, value, where in [(0, 'attn_pre_softmax', score_value, score_mask),
                                     (1, 'k', k_value, k_mask), (1, 'v', v_value, v_mask)]:
        cache['transformer_output']['block_outputs'][layer]['attn_output'][name] = value
        mask['transformer_output']['block_outputs'][layer]['attn_output'][name] = where
    return model.call_with_all_aux(examples=x, labels=y, key=KEY, cache=cache, cache_mask=mask)


full_fn = eqx.filter_jit(jax.vmap(forward, in_axes=(None, 0, 0, 0, None, 0, None, 0, None)))


def make_author_reference(hard_block=False):
    opts = main_utils.get_opts_from_json_file(CFG)
    if hard_block:
        opts.opto_disallow_prev_token_heads = ['0:0']
        opts.opto_disallow_prev_token_only_label = True
    fn = opto.make_fn_from_opts(opts)
    call = lambda model, x, y: fn(model=model, x=x, y=y, key=KEY)
    return eqx.filter_jit(jax.vmap(call, in_axes=(None, 0, 0)))


def native(model, x, y):
    count = len(x)
    score = jnp.zeros((count, 1, 8, 5, 5), jnp.float32)
    kv = jnp.zeros((count, 1, 8, 5, 8), jnp.float32)
    return full_fn(model, x, y, score, jnp.asarray(ZS), kv, jnp.asarray(ZKV), kv, jnp.asarray(ZKV))


def suppress(model, x, y, rn, code):
    score = attn(rn, 0)['attn_pre_softmax']
    mask = PREV
    if code in ('P1', 'S'):
        value = score - SETTINGS['primary_delta']
    elif code == 'P2':
        value = score - SETTINGS['sensitivity_delta']
    elif code == 'PB':
        value = jnp.full_like(score, SETTINGS['hard_block_value'])
    elif code == 'H6':
        value, mask = score - SETTINGS['primary_delta'], H6
    elif code == 'SELF':
        value, mask = score - SETTINGS['primary_delta'], SELF
    elif code == 'H6R':
        # Match relative previous mass, using only native attention and fixed dose.
        # Stable odds formulation avoids p6=1 rounding and p0'/p0 zero division.
        value, mask = score, H6
        a = attn(rn, 0)['attn_scores']
        for row, col in [(1, 0), (3, 2)]:
            p0 = a[:, 0, 0, row, col]
            e = jnp.exp(-SETTINGS['primary_delta'])
            log_r = jnp.log(e) - jnp.log1p(p0 * (e - 1))
            valid_others = np.array([j <= row and j != col for j in range(5)])
            others = jnp.where(jnp.asarray(valid_others), score[:, 0, 6, row, :], -jnp.inf)
            other_lse = jax.scipy.special.logsumexp(others, axis=-1)
            log_odds = score[:, 0, 6, row, col] - other_lse
            log_target = jax.nn.log_sigmoid(log_odds) + log_r
            # If log_r==0 the mass is unchanged; avoid log(0) at saturation.
            bounded = jnp.minimum(log_target, -1e-7)
            log_one_minus = jnp.where(bounded < -jnp.log(2.),
                          jnp.log1p(-jnp.exp(bounded)), jnp.log(-jnp.expm1(bounded)))
            replacement = other_lse + bounded - log_one_minus
            replacement = jnp.where(log_r == 0, score[:, 0, 6, row, col], replacement)
            value = value.at[:, 0, 6, row, col].set(replacement)
    else:
        raise ValueError(code)
    return full_fn(model, x, y, value, jnp.asarray(mask), attn(rn, 1)['k'], jnp.asarray(ZKV),
                   attn(rn, 1)['v'], jnp.asarray(ZKV))


def path_result(model, x, y, rn, rs, code):
    if code == 'N':
        return rn
    if code == 'S':
        return rs
    scores = attn(rs, 0)['attn_pre_softmax']
    sm = PREV if code in ('KB', 'VB') else ZS
    ksource = rn if code == 'KB' else rs
    vsource = rn if code == 'VB' else rs
    km = LABEL_KV if code in ('K', 'KB', 'KV') else ZKV
    vm = LABEL_KV if code in ('V', 'VB', 'KV') else ZKV
    return full_fn(model, x, y, scores, jnp.asarray(sm), attn(ksource, 1)['k'], jnp.asarray(km),
                   attn(vsource, 1)['v'], jnp.asarray(vm))


def one_metrics(r, y, rn, projection):
    z = r['out'][-1]
    target = y[-1]
    other = y[0] + y[1] - target
    full_pred = jnp.argmax(z).astype(jnp.int32)
    choose0 = (z[y[0]] > z[y[1]]) | ((z[y[0]] == z[y[1]]) & (y[0] < y[1]))
    pair_pred = jnp.where(choose0, y[0], y[1]).astype(jnp.int32)
    margin = z[target] - z[other]
    lse = jax.scipy.special.logsumexp(z)
    l1, l2 = attn(r, 0), attn(r, 1)
    ti, oi = jnp.where(y[0] == target, 1, 3), jnp.where(y[0] == target, 3, 1)
    qs, qa = l2['attn_pre_softmax'][0, :, 4, :], l2['attn_scores'][0, :, 4, :]
    label_delta = (l1['values_pre_proj'][jnp.array([1, 3])] -
                   attn(rn, 0)['values_pre_proj'][jnp.array([1, 3])]).reshape(2, 8, 8)
    delta = jnp.take(label_delta, jnp.array([0, 6]), axis=1)
    proj = jnp.take(projection.reshape(64, 8, 8), jnp.array([0, 6]), axis=1)
    head_delta = jnp.einsum('lhk,dhk->hld', delta, proj)
    prev = lambda a: jnp.stack([a[0, :, 1, 0], a[0, :, 3, 2]], axis=-1)
    selfa = lambda a: jnp.stack([a[0, :, 1, 1], a[0, :, 3, 3]], axis=-1)
    return dict(full_loss=lse-z[target], pair_loss=jax.nn.softplus(-margin), output_margin=margin,
                full_acc=(full_pred == target).astype(jnp.float32),
                pair_acc=(pair_pred == target).astype(jnp.float32),
                full_pred=full_pred, pair_pred=pair_pred,
                outside_pair=((full_pred != y[0]) & (full_pred != y[1])).astype(jnp.float32),
                full_prob=jnp.exp(z[target]-lse), pair_prob=jax.nn.sigmoid(margin),
                context_prob=jnp.exp(jnp.logaddexp(z[y[0]], z[y[1]])-lse),
                target_logit=z[target], other_logit=z[other], full_logsumexp=lse,
                l2_qk_margin=qs[:, ti]-qs[:, oi], l2_attention_margin=qa[:, ti]-qa[:, oi],
                l1_previous_attention=prev(l1['attn_scores']), l1_self_attention=selfa(l1['attn_scores']),
                l1_previous_score=prev(l1['attn_pre_softmax']),
                head_label_delta_norm=jnp.linalg.norm(head_delta, axis=-1))


metric_fn = eqx.filter_jit(jax.vmap(one_metrics, in_axes=(0, 0, 0, None)))


def metrics_numpy(r, y, rn, model):
    m = metric_fn(r, y, rn, model.transformer.blocks[0].attn.proj.weight)
    out = {k: np.asarray(v) for k, v in m.items()}
    # Independent CPU calculation from full logits in every batch, not just means.
    z = np.asarray(r['out'][:, -1], np.float64)
    yy = np.asarray(y)
    pred = z.argmax(axis=1).astype(np.int32)
    assert np.array_equal(pred, out['full_pred'])
    assert np.array_equal((pred == yy[:, -1]).astype(np.float32), out['full_acc'])
    a, b = z[np.arange(len(z)), yy[:, 0]], z[np.arange(len(z)), yy[:, 1]]
    pair = np.where((a > b) | ((a == b) & (yy[:, 0] < yy[:, 1])), yy[:, 0], yy[:, 1])
    assert np.array_equal(pair, out['pair_pred'])
    cpu_lse = np.log(np.exp(z-z.max(axis=1)[:, None]).sum(axis=1)) + z.max(axis=1)
    cpu_loss = cpu_lse-z[np.arange(len(z)), yy[:, -1]]
    assert np.max(np.abs(cpu_loss-out['full_loss'])) <= 1e-4
    assert all(np.isfinite(v).all() for v in out.values())
    return out
