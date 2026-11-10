"""Copy and re-audit the five registered exp036 HDFs; never generate new inputs."""
from plan import *
import shutil


def prepare():
    import numpy as np
    import h5py
    from itertools import combinations
    verify_code()
    spec = read_json(WORK / 'input_spec.json')
    target = OUTPUTS / 'data'
    target.mkdir(parents=True, exist_ok=True)
    for name in SETTINGS['data_sets']:
        filename = name + '.h5'
        expected = spec['files']['frozen_data/' + filename]['sha256']
        source, dest = ASSETS / 'frozen_data' / filename, target / filename
        if sha(source) != expected:
            raise RuntimeError('registered input changed: ' + filename)
        if not dest.exists():
            temp = dest.with_suffix('.partial')
            shutil.copyfile(source, temp)
            temp.replace(dest)
        if sha(dest) != expected:
            raise RuntimeError('preserved data changed: ' + filename)
    features_path = ASSETS / 'data/omniglot_features_norotate.h5'
    assert sha(features_path) == spec['files']['data/omniglot_features_norotate.h5']['sha256']
    with h5py.File(features_path, 'r') as f:
        features = np.asarray(f['resnet18/224/feat'][:1600, :5], np.float32)
        assert np.array_equal(f['original_class'][:1600], np.arange(1600))
        assert np.all(f['augmentation'][:1600] == 0)
    lookup = {hashlib.sha256(features[c, i].tobytes()).digest(): (c, i)
              for c in range(1600) for i in range(5)}
    assert len(lookup) == 8000
    complete, visible, records = {}, {}, {}
    for name, cfg in SETTINGS['data_sets'].items():
        complete[name], visible[name] = set(), set()
        tasks = {}
        with h5py.File(target / (name + '.h5'), 'r') as f:
            for task in SETTINGS['fresh_tasks']:
                x, y, ids = (np.asarray(f[task][key]) for key in ['examples', 'labels', 'identities'])
                assert x.shape == (cfg['n'], 3, 512) and y.shape == (cfg['n'], 3)
                actual = np.asarray([lookup[hashlib.sha256(v.tobytes()).digest()]
                                     for v in x.reshape(-1, 512)], np.int32).reshape(-1, 3, 2)
                assert np.array_equal(actual, ids)
                cls = ids[:, :, 0]
                same, hits = cls[:, :2] == cls[:, -1, None], y[:, :2] == y[:, -1, None]
                assert np.all(cls[:, 0] != cls[:, 1]) and np.all(y[:, 0] != y[:, 1])
                assert np.all(hits.sum(1) == 1)
                if task == 'iwl_copy_avail':
                    assert np.all(same.sum(1) == 0) and np.array_equal(y[:, -1], cls[:, -1])
                else:
                    assert np.all(same.sum(1) == 1) and np.array_equal(y[:, :2][same], y[:, -1])
                    if task == 'train_eval':
                        assert np.array_equal(y, cls)
                    elif task == 'flip_icl':
                        assert np.array_equal(y[:, :2], cls[:, 1::-1])
                    else:
                        assert np.all(np.sort(y[:, :2], axis=1) == [0, 1])
                for a, b in zip(x, y):
                    complete[name].add(hashlib.sha256(a.tobytes() + b.tobytes()).digest())
                    visible[name].add(hashlib.sha256(a.tobytes() + b[:2].tobytes()).digest())
                tasks[task] = dict(rows=len(y), array_sha256=hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest())
        records[name] = dict(sha256=sha(target / (name + '.h5')), tasks=tasks)
        (target / (name + '.h5')).chmod(0o444)
    overlaps = {a + '/' + b: dict(full=len(complete[a] & complete[b]), visible=len(visible[a] & visible[b]))
                for a, b in combinations(SETTINGS['data_sets'], 2)}
    passed = all(v['full'] == v['visible'] == 0 for v in overlaps.values())
    result = dict(passed=passed, data=records, overlaps=overlaps, resampling=False,
                  code_manifest_sha256=code_identity())
    write_json(target / 'identity_audit.json', result)
    if not passed:
        raise RuntimeError('registered purpose independence failed')
    write_json(target / 'complete.json', result)
    return result
