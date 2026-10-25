"""New fixed purposes; author samplers and exact identity audit, no resampling."""
from common import *
from collections import Counter
from itertools import combinations

ORDER = ('train_eval', 'iwl_copy_avail', 'flip_icl', 'icl', 'pure_iwl')
CONFIGS = {
    'train_eval': dict(burstiness=1, no_support=False, unique_rest=False, assign_query_label_random=False, fs_relabel=None),
    'iwl_copy_avail': dict(burstiness=0, no_support=True, unique_rest=True, assign_query_label_random=True, fs_relabel=None),
    'flip_icl': dict(burstiness=1, no_support=False, unique_rest=False, assign_query_label_random=False, fs_relabel='flip'),
    'icl': dict(burstiness=1, no_support=False, unique_rest=False, assign_query_label_random=False, fs_relabel='01')}


def row_key(x, y):
    return hashlib.sha256(np.asarray(x, np.float32).tobytes() + np.asarray(y, np.int32).tobytes()).hexdigest()


def visible_key(x, y):
    return row_key(x, y[:2])


def prepare():
    verify_inputs()
    creation = SETTINGS.get('data_generator_manifest_sha256')
    if creation is not None:
        assert sha(WORK / 'data_creation_manifest.json') == creation
        preserved = SETTINGS['preserved_data_sha256']
        assert set(preserved) == {name + '.h5' for name in SETTINGS['data_sets'] if name != 'mechanism_discovery'}
        for filename, expected_sha in preserved.items():
            path = OUT / 'data' / filename
            if not path.is_file() or sha(path) != expected_sha:
                raise RuntimeError('preserved generated data missing or changed; no new draws under old identity: ' + filename)
    with h5py.File(ASSETS / 'data/omniglot_features_norotate.h5', 'r') as f:
        features = np.asarray(f['resnet18/224/feat'][:1600, :5], np.float32)
        assert np.array_equal(f['original_class'][:1600], np.arange(1600))
        assert np.all(f['augmentation'][:1600] == 0)
    assert features.shape == (1600, 5, 512)
    lookup = {}
    for c in range(1600):
        for s in range(5):
            key = hashlib.sha256(features[c, s].tobytes()).digest()
            assert key not in lookup
            lookup[key] = (c, s)
    audit, rows, visible_rows = {}, {}, {}
    for name, setting in SETTINGS['data_sets'].items():
        path = OUT / 'data' / (name + '.h5')
        path.parent.mkdir(exist_ok=True)
        metadata = dict(name=name, **setting,
                        code_manifest_sha256=(SETTINGS['discovery_generator_manifest_sha256'] if name == 'mechanism_discovery' else SETTINGS['data_generator_manifest_sha256']),
                        sampler_sha256=sha(ASSETS / 'source/samplers.py'),
                        feature_sha256=sha(ASSETS / 'data/omniglot_features_norotate.h5'),
                        task_order=list(ORDER), configuration=CONFIGS)
        if name == 'mechanism_discovery':
            metadata['generator_version'] = SETTINGS['discovery_generator_version']
        if path.exists():
            with h5py.File(path, 'r') as f:
                assert json.loads(f.attrs['metadata']) == metadata
                xy = {t: (f[t]['examples'][:], f[t]['labels'][:]) for t in TASKS}
                saved_ids = {t: f[t]['identities'][:] for t in TASKS}
        else:
            xy, saved_ids = {}, None
            keys = jax.random.split(jax.random.PRNGKey(setting['seed']), 5)
            for i, task in enumerate(ORDER[:4]):
                cfg = dict(CONFIGS[task])
                relabel = cfg.pop('fs_relabel')
                if relabel == '01':
                    relabel = jnp.array([[0, 1]])
                random = cfg.pop('assign_query_label_random')
                cs = partial(samplers.get_constant_burst_seq_idxs, classes=jnp.arange(1600),
                    class_distr=jnp.ones(1600) / 1600, num_seqs=setting['n'], context_len=2,
                    distractor=True, **cfg)
                es = partial(samplers.get_exemplar_inds, allowed_inds=jnp.arange(5),
                    match_query_and_distractors=False)
                sampler = samplers.make_data_sampler(cs, es, fs_relabel=relabel,
                    noise_scale=0.0, assign_query_label_random=random)
                batch = sampler(keys[i], jnp.asarray(features))
                xy[task] = (np.asarray(batch['examples'], np.float32),
                            np.asarray(batch['labels'], np.int32))
        identity, task_audit, rows[name], visible_rows[name] = {}, {}, {}, {}
        for task, (x, y) in xy.items():
            assert x.shape == (setting['n'], 3, 512) and y.shape == (setting['n'], 3)
            ids = np.asarray([lookup[hashlib.sha256(v.tobytes()).digest()]
                              for v in x.reshape(-1, 512)], np.int32).reshape(-1, 3, 2)
            if saved_ids is not None:
                assert np.array_equal(ids, saved_ids[task])
            classes = ids[:, :, 0]
            same = classes[:, :2] == classes[:, -1, None]
            target = y[:, :2] == y[:, -1, None]
            assert np.all(classes[:, 0] != classes[:, 1]) and np.all(y[:, 0] != y[:, 1])
            assert np.all(target.sum(1) == 1)
            if task == 'iwl_copy_avail':
                assert np.all(same.sum(1) == 0) and np.array_equal(y[:, -1], classes[:, -1])
                assert np.array_equal(y[:, :2][~target], classes[:, :2][~target])
            else:
                assert np.all(same.sum(1) == 1) and np.array_equal(y[:, :2][same], y[:, -1])
                if task == 'train_eval':
                    assert np.array_equal(y, classes)
                elif task == 'flip_icl':
                    assert np.array_equal(y[:, :2], classes[:, 1::-1])
                    assert np.all(y[:, -1] != classes[:, -1])
                else:
                    assert np.all(np.sort(y[:, :2], axis=1) == np.array([0, 1]))
            rr = [row_key(a, b) for a, b in zip(x, y)]
            identity[task], rows[name][task] = ids, set(rr)
            visible_rows[name][task] = {visible_key(a, b) for a,b in zip(x,y)}
            task_audit[task] = dict(n=len(x), unique_rows=len(set(rr)),
                duplicate_rows=len(rr) - len(set(rr)), query_classes=len(set(classes[:, -1].tolist())),
                array_sha256=hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest())
        if not path.exists():
            temporary = path.with_name(path.name + '.partial')
            with h5py.File(temporary, 'w') as f:
                f.attrs['metadata'] = json.dumps(metadata, sort_keys=True)
                for task, (x, y) in xy.items():
                    g = f.create_group(task)
                    g.create_dataset('examples', data=x, compression='gzip', compression_opts=1)
                    g.create_dataset('labels', data=y)
                    g.create_dataset('identities', data=identity[task])
            os.link(temporary, path)
            temporary.unlink()
        audit[name] = dict(sha256=sha(path), tasks=task_audit)
    overlaps, visible_overlaps = {}, {}
    for left, right in combinations(SETTINGS['data_sets'], 2):
        l = set().union(*rows[left].values())
        r = set().union(*rows[right].values())
        overlaps[left + '/' + right] = len(l & r)
        visible_overlaps[left + '/' + right] = len(set().union(*visible_rows[left].values())
                                                  & set().union(*visible_rows[right].values()))
    active = SETTINGS.get('active_data_sets', list(SETTINGS['data_sets']))
    if SETTINGS.get('analysis_scope') == 'single_seed_pilot':
        assert active == ['mechanism_confirm','protection','monitor','confirm']
    assert len(active) == len(set(active)) and all(n in SETTINGS['data_sets'] for n in active)
    active_keys = [left + '/' + right for left, right in combinations(active, 2)]
    passed = all(overlaps[k] == 0 and visible_overlaps[k] == 0 for k in active_keys)
    write_json(OUT / 'data/identity_audit.json', dict(passed=passed, data=audit,
               exact_sequence_overlap=overlaps, visible_model_input_overlap=visible_overlaps, resampling=False,
               active_data_sets=active, active_independence_pairs=active_keys,
               inactive_data_sets=[n for n in SETTINGS['data_sets'] if n not in active],
               inactive_policy='preserved and audited but not used for selection, protection or evaluation',
               identity_definition='ordered float32 examples and int32 labels including target'))
    if not passed:
        raise RuntimeError('prespecified data independence gate failed; no filtering/resampling')


if __name__ == '__main__':
    prepare()
