"""GPU-worker helpers. The controller and local preparation never import this."""
from plan import *
from functools import partial
from importlib.metadata import version as distribution_version
import sys
import time
import platform

OUT = OUTPUTS
DATA = OUT / 'data'
SPEC = read_json(WORK / 'input_spec.json')
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


def arrays(tree):
    return [v for v in jax.tree_util.tree_leaves(tree) if eqx.is_array(v)]


def sync(tree):
    for value in arrays(tree):
        if hasattr(value, 'block_until_ready'):
            value.block_until_ready()
    return tree


def exact(a, b):
    aa, bb = arrays(a), arrays(b)
    return len(aa) == len(bb) and all(x.shape == y.shape and x.dtype == y.dtype
        and np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(aa, bb))


def maxdiff(a, b):
    aa, bb = arrays(a), arrays(b)
    if len(aa) != len(bb):
        raise ValueError('different array counts')
    return max([float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))))
                for x, y in zip(aa, bb) if x.size] + [0.0])


def state_digest(tree):
    h = hashlib.sha256()
    for value in arrays(tree):
        a = np.asarray(value)
        h.update(str((a.shape, str(a.dtype))).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def verify_inputs(require_gpu=True):
    verify_code()
    inputs = {}
    for name, record in SPEC['files'].items():
        actual = sha(ASSETS / name)
        if actual != record['sha256']:
            raise RuntimeError('scientific asset identity: ' + name)
        inputs[name] = actual
    versions = {k: distribution_version(k) for k in SPEC['pinned_versions'] if k != 'jaxlib_distribution'}
    versions['jaxlib_distribution'] = distribution_version('jaxlib')
    for name, expected in SPEC['pinned_versions'].items():
        if versions[name] != expected:
            raise RuntimeError('incompatible pinned version: ' + name)
    if require_gpu and jax.default_backend() != 'gpu':
        raise RuntimeError('CUDA required; no CPU fallback')
    if jax.config.jax_enable_x64 or jax.config.jax_default_matmul_precision is not None:
        raise RuntimeError('author FP32/default matmul required')
    for module in [main, main_utils, samplers, opto, sys.modules['models']]:
        if Path(module.__file__).resolve().parent != (ASSETS / 'source').resolve():
            raise RuntimeError('unintended author source import')
    return dict(inputs=inputs, versions=versions, device=[str(d) for d in jax.devices()],
                device_kind=[d.device_kind for d in jax.devices()], python=sys.version,
                platform=platform.platform(), code_manifest_sha256=code_identity())


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
    for value in jax.tree_util.tree_leaves(model, is_leaf=lambda x: isinstance(x, eqx.nn.Dropout)):
        if isinstance(value, eqx.nn.Dropout) and value.p != 0:
            raise RuntimeError('unexpected model dropout')
    optimizer = main_utils.get_optimizer_from_opts(load_opts(seed))
    ck = dict(iter=-1, model=model, opt_state=optimizer.init(eqx.filter(model, eqx.is_array)),
              seeds=dict(eval_model_seed=KEY, train_data_seed=KEY, train_model_seed=KEY))
    return ck, optimizer


def initial_checkpoint(seed):
    ck, _ = template(seed)
    dk, mk = jax.random.split(jax.random.PRNGKey(seed['train']))
    _, ek = jax.random.split(jax.random.PRNGKey(load_opts(seed).eval_seed))
    ck.update(iter=0, seeds=dict(train_data_seed=dk, train_model_seed=mk, eval_model_seed=ek))
    return ck


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


def base_checkpoint(seed):
    return load_checkpoint(OUT / seed['id'] / 'prefix' / '00004400000.eqx', seed)


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
    with h5py.File(DATA / (name + '.h5'), 'r') as f:
        data = {t: (np.asarray(f[t]['examples'], np.float32), np.asarray(f[t]['labels'], np.int32)) for t in TASKS}
        ids = {t: np.asarray(f[t]['identities'], np.int32) for t in TASKS}
    return data, ids


def save_npz(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.npz')
    np.savez_compressed(temp, **record)
    temp.replace(path)


def embedding_leaves(model):
    return (model.example_embed.weight, model.example_embed.bias, model.label_embed.weight)
