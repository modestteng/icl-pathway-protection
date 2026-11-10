"""Cloud runtime using immutable author source and approved scientific assets."""
from pathlib import Path
from functools import partial
import hashlib
import json
import os
import sys
import time
from importlib.metadata import version as distribution_version

WORK = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get('ICL031_ASSETS', str(WORK / 'assets'))).resolve()
OUT = WORK / 'outputs'
OUT.mkdir(exist_ok=True)
SETTINGS = json.loads((WORK / 'settings.json').read_text())
SPEC = json.loads((WORK / 'input_spec.json').read_text())
sys.path.insert(0, str(ASSETS / 'source'))

import numpy as np
import h5py
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import main
import main_utils
import samplers
import opto

TASKS = tuple(SETTINGS['fresh_tasks'])
BS = SETTINGS['train_bs']
KEY = jax.random.PRNGKey(SETTINGS['forward_seed'])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def arrays(tree):
    return [v for v in jax.tree_util.tree_leaves(tree) if eqx.is_array(v)]


def sync(tree):
    for v in arrays(tree):
        if hasattr(v, 'block_until_ready'):
            v.block_until_ready()
    return tree


def maxdiff(a, b):
    aa, bb = arrays(a), arrays(b)
    if len(aa) != len(bb):
        raise ValueError('different array counts')
    return max([float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))))
                for x, y in zip(aa, bb) if x.size] + [0.0])


def exact(a, b):
    aa, bb = arrays(a), arrays(b)
    return len(aa) == len(bb) and all(np.array_equal(np.asarray(x), np.asarray(y))
                                    for x, y in zip(aa, bb))


def verify_inputs():
    manifest = json.loads((WORK / 'code_manifest.json').read_text())
    for name, digest in manifest.items():
        if sha(WORK / name) != digest:
            raise ValueError('changed sealed code: ' + name)
    inputs = {}
    for name, record in SPEC['files'].items():
        path = ASSETS / name
        digest = sha(path)
        if digest != record['sha256']:
            raise ValueError('scientific asset identity: ' + name)
        inputs[name] = digest
    versions = dict(jax=jax.__version__, jaxlib=__import__('jaxlib').__version__,
                    jaxlib_distribution=distribution_version('jaxlib'),
                    equinox=eqx.__version__, optax=optax.__version__,
                    numpy=np.__version__, h5py=h5py.__version__)
    for name, version in [('jax', '0.4.26'), ('jaxlib_distribution','0.4.26+cuda12.cudnn89'),
                          ('equinox', '0.11.4'), ('optax', '0.2.2'),
                          ('numpy','1.26.4'),('h5py','3.11.0')]:
        if versions[name] != version:
            raise RuntimeError('incompatible pinned version: ' + name)
    if jax.default_backend() != 'gpu':
        raise RuntimeError('CUDA GPU required; no automatic CPU/Mac fallback')
    if jax.config.jax_enable_x64 or jax.config.jax_default_matmul_precision is not None:
        raise RuntimeError('author float32/default matmul semantics required')
    imported = {name: sha(ASSETS / 'source' / name)
                for name in SPEC['source_files']}
    for module in [main, main_utils, samplers, opto]:
        if Path(module.__file__).resolve().parent != (ASSETS / 'source').resolve():
            raise RuntimeError('source imported from unintended location')
    asset_manifest_path = ASSETS / 'assets_manifest.json'
    if sha(asset_manifest_path)!=SPEC['registered_asset_manifest_sha256']:
        raise RuntimeError('asset export differs from registered preparation')
    if sha(ASSETS/SPEC['replay_checkpoint']['to'])!=SPEC['replay_checkpoint']['sha256']:
        raise RuntimeError('replay reference identity changed')
    asset_manifest = json.loads(asset_manifest_path.read_text())
    for name, record in asset_manifest.items():
        if sha(ASSETS / name) != record['sha256']:
            raise RuntimeError('exported asset manifest mismatch: ' + name)
    result = dict(inputs=inputs, source=imported, versions=versions, assets_root=str(ASSETS),
                  assets_manifest_sha256=sha(asset_manifest_path),
                  code_manifest_sha256=sha(WORK / 'code_manifest.json'),
                  device=[str(d) for d in jax.devices()],
                  default_matmul_precision=None, enable_x64=False)
    old = OUT / 'provenance.json'
    if old.exists():
        prior = json.loads(old.read_text())
        new_inputs = prior.pop('new_inputs', {})
        if prior != result:
            raise RuntimeError('provenance changed; archive attempt before resume')
        for name, digest in new_inputs.items():
            if sha(WORK / name) != digest:
                raise RuntimeError('fresh data content changed: ' + name)
        if new_inputs:
            result['new_inputs'] = new_inputs
    write_json(old, result)
    return result


def load_opts(seed=None):
    opts = main_utils.get_opts_from_json_file(ASSETS / 'config.json')
    opts.data_file = str(ASSETS / 'data' / 'omniglot_features_reordered.h5')
    if seed is not None:
        opts.init_seed, opts.train_seed = seed['init'], seed['train']
    assert (opts.depth, opts.num_heads, opts.d_model) == (2, 8, 64)
    assert opts.mlp_ratio is None and opts.model_output_classes == 12800
    assert opts.lr == 1e-5 and opts.optimizer == 'adam' and opts.weight_decay == 0
    assert opts.train_bs == BS and opts.train_microbs == BS
    assert opts.pos_embedding_type == 'ape' and not opts.match_query_and_distractors
    return opts


def template(seed=None):
    opts = load_opts(seed)
    model = main_utils.get_model_from_opts(opts, (512,))
    for v in jax.tree_util.tree_leaves(model, is_leaf=lambda x: isinstance(x, eqx.nn.Dropout)):
        if isinstance(v, eqx.nn.Dropout) and v.p != 0:
            raise RuntimeError('unexpected stochastic model; audit key semantics')
    optimizer = main_utils.get_optimizer_from_opts(load_opts(seed))
    ck = dict(iter=-1, model=model, opt_state=optimizer.init(eqx.filter(model, eqx.is_array)),
              seeds=dict(eval_model_seed=KEY, train_data_seed=KEY, train_model_seed=KEY))
    return ck, optimizer


def load_checkpoint(path, seed=None):
    tmpl, _ = template(seed)
    ck = eqx.tree_deserialise_leaves(path, tmpl)
    assert int(ck['iter']) // BS == int(np.asarray(ck['opt_state'][0].count))
    assert all(np.isfinite(np.asarray(v)).all() for v in arrays(ck))
    return ck


def save_checkpoint(path, ck):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sync(ck)
    tmp = path.with_name(path.name + '.tmp')
    eqx.tree_serialise_leaves(tmp, ck)
    tmp.replace(path)
    return sha(path)


def initial_checkpoint(seed):
    if seed['id'] == 's5':
        return load_checkpoint(ASSETS / 'checkpoints' / '00000000000.eqx', seed)
    ck, _ = template(seed)
    opts = load_opts(seed)
    dk, mk = jax.random.split(jax.random.PRNGKey(seed['train']))
    _, ek = jax.random.split(jax.random.PRNGKey(opts.eval_seed))
    ck.update(iter=0, seeds=dict(train_data_seed=dk, train_model_seed=mk, eval_model_seed=ek))
    return ck


def base_checkpoint(seed):
    if seed['id'] == 's5':
        path = ASSETS / 'checkpoints' / '00004400000.eqx'
    else:
        path = OUT / seed['id'] / 'prefix' / '00004400000.eqx'
    ck = load_checkpoint(path, seed)
    assert int(ck['iter']) == SETTINGS['start_stage']
    return ck


def make_sampler(seed):
    opts = load_opts(seed)
    data = main_utils.get_data_from_opts(opts)
    splits = main_utils.get_splits_from_opts(opts, data.shape)
    ranks = jnp.arange(1, opts.class_split[0] + 1, dtype=jnp.float32)
    probability = 1 / ranks ** opts.zipf_alpha
    probability /= probability.sum()
    substrates = [partial(samplers.get_constant_burst_seq_idxs,
        classes=splits['class']['train'], class_distr=probability, num_seqs=BS,
        context_len=opts.train_context_len, burstiness=opts.pt_burstiness[i],
        distractor=main.smart_index(opts.pt_distract, i, 1),
        no_support=main.smart_index(opts.pt_no_support, i, 0),
        unique_rest=main.smart_index(opts.pt_unique_rest, i, 0))
        for i in range(len(opts.mixing_coeffs))]
    cs = partial(samplers.get_mixed_seq_idxs, mix_probabilities=jnp.array(opts.mixing_coeffs),
                 mix_substrate_fns=substrates)
    es = partial(samplers.get_exemplar_inds, allowed_inds=splits['exemplar']['train'],
                 match_query_and_distractors=opts.match_query_and_distractors)
    native = samplers.make_data_sampler(cs, es, fs_relabel=splits['relabeling']['train'],
        noise_scale=opts.noise_scale_train, assign_query_label_random=opts.assign_query_label_random)
    compiled = jax.jit(native)
    def sample(key):
        return compiled(key, data)
    sample.native, sample.data = native, data
    return sample


def load_dataset(name):
    path = OUT / 'data' / (name + '.h5')
    with h5py.File(path, 'r') as f:
        data = {task: (np.asarray(f[task]['examples'], np.float32),
                       np.asarray(f[task]['labels'], np.int32)) for task in TASKS}
        ids = {task: np.asarray(f[task]['identities'], np.int32) for task in TASKS}
    return data, ids


def save_npz(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.npz')
    np.savez_compressed(tmp, **record)
    tmp.replace(path)


def code_identity():
    return sha(WORK / 'code_manifest.json')


def provenance():
    return json.loads((OUT / 'provenance.json').read_text())


def embedding_leaves(model):
    return (model.example_embed.weight, model.example_embed.bias, model.label_embed.weight)
