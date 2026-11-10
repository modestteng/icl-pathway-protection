"""Audit the exact code-only release, including the staged Git snapshot."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from collections import Counter
from numbers import Integral
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
POLICY = 'reproducibility/release_policy.json'
ALLOWED_EXTENSIONS = {'.py', '.json', '.md', '.txt'}
EXCLUDED_PARTS = {'assets', 'outputs', 'results', 'final_results', 'review_tables',
                  'summary', 'checkpoints', 'logs', 'work', 'manuscript', 'tmp',
                  'backup', 'backups', '__pycache__', 'source'}
PRIVATE = {
    'private_path': re.compile(r'/(?:' + '|'.join(('Users', 'home', 'root')) + r')/|[A-Z]:[\\/]'),
    'email': re.compile(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}'),
    'credential': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}'),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def disk_files():
    return {p.relative_to(ROOT).as_posix(): p.read_bytes()
            for p in ROOT.rglob('*') if p.is_file()
            and p.relative_to(ROOT).parts[0] != '.git'}


def index_files():
    rows = subprocess.check_output(['git', 'ls-files', '--stage', '-z'], cwd=ROOT).split(b'\0')
    files = {}
    for row in rows:
        if not row:
            continue
        meta, path = row.split(b'\t', 1)
        mode, object_id, stage = meta.split()
        if mode != b'100644' or stage != b'0':
            raise ValueError('Unmerged, executable, symlink or submodule entry: ' + path.decode())
        files[path.decode()] = subprocess.check_output(['git', 'cat-file', 'blob', object_id], cwd=ROOT)
    return files


def check_paths(files, policy, errors):
    for name, data in files.items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or set(p.parts) & EXCLUDED_PARTS:
            errors.append('Excluded or unsafe path: ' + name)
        if p.suffix not in ALLOWED_EXTENSIONS and name != '.gitignore':
            errors.append('Non-code/configuration file: ' + name)
        if len(data) > policy['max_file_bytes']:
            errors.append('Per-file budget exceeded: ' + name)
        try:
            text = data.decode('utf-8')
            if p.suffix == '.py':
                ast.parse(text, filename=name)
            elif p.suffix == '.json':
                json.loads(text)
            for kind, pattern in PRIVATE.items():
                if pattern.search(text):
                    errors.append(kind + ': ' + name)
        except (UnicodeError, SyntaxError, ValueError) as exc:
            errors.append('Invalid UTF-8, Python or JSON: ' + name + ': ' + str(exc))
    if len(files) > policy['max_files']:
        errors.append('File count budget exceeded')
    if sum(map(len, files.values())) > policy['max_total_bytes']:
        errors.append('Total release budget exceeded')


def check_seed_identities(load, registry):
    """Compare complete group/architecture/ID/init/train records with configs."""
    def architecture(config):
        return f"{config['depth']}L/{config['d_model']}"

    def rows(group, architectures, seeds):
        ids = [seed['id'] for seed in seeds]
        assert len(ids) == len(set(ids)), 'Duplicate seed ID within group: ' + group
        assert len(architectures) == len(set(architectures)), 'Duplicate architecture: ' + group
        assert architectures and seeds, 'Empty seed group: ' + group
        return {(group, arch, seed['id'], seed['init'], seed['train'])
                for arch in architectures for seed in seeds}

    group_ids = [group['id'] for group in registry['groups']]
    assert len(group_ids) == len(set(group_ids)), 'Duplicate experiment group'
    groups = {group['id']: group for group in registry['groups']}
    owners = {
        'primary': 'exp_033_random_path_only_protection',
        'independent_replication': 'exp_036_multiseed_path_protection',
        'two_layer_scale': 'exp_040_scale_generalization_v2',
        'fixed_start_and_transfer': 'exp_042_seed_and_horizon',
    }
    assert set(groups) == set(owners) | {'four_layer_depth'}, 'Unexpected seed groups'
    baseline = architecture(load('experiments/exp_002_main_reproduction/final_00020000000/original_config.json'))
    expected = set()
    configured = set()
    count = 0
    for key, group in groups.items():
        expected.update(rows(key, group['architectures'], group['seeds']))
        count += len(group['seeds'])
        if key in owners:
            config = load('experiments/' + owners[key] + '/settings.json')
            architectures = ([architecture(config['architectures'][name]) for name in ('base', 'width')]
                             if key == 'two_layer_scale' else [baseline])
            configured.update(rows(key, architectures, config['training_seeds']))
        else:
            depth_ids = set()
            for family in ('exp_054_uncapped_five_seeds', 'exp_055_uncapped_four_seeds'):
                protocol = load('experiments/' + family + '/protocol.json')
                for sid, seed in protocol['seeds'].items():
                    assert sid == seed['id'], 'Protocol key differs from seed ID: ' + sid
                    assert sid not in depth_ids, 'Duplicate four-layer ID: ' + sid
                    depth_ids.add(sid)
                configured.update(rows(key, [architecture(protocol['architecture'])],
                                       list(protocol['seeds'].values())))
            assert len(depth_ids) == 8, 'Four-layer group must contain eight manuscript IDs'
    assert expected == configured, 'Seed identity mismatch (group, architecture, ID, init, train): ' + str(sorted(expected ^ configured))
    assert count == 20 and len(expected) == 25, 'Unexpected manuscript seed counts'
    return count, len(expected)


def check_seeds(files):
    load = lambda name: json.loads(files[name])
    registry = load('reproducibility/manuscript_seed_registry.json')
    assert registry['schema'] == 'icl_manuscript_seed_registry_v2'
    identity_count, trajectories = check_seed_identities(load, registry)
    groups = {g['id']: g for g in registry['groups']}
    guide = files['reproducibility/SEEDS.md'].decode()
    records = 0
    for key, group in groups.items():
        for seed in group['seeds']:
            row = '| {} | {} | {} | {} | {} |'.format(
                key, ', '.join(group['architectures']), seed['id'], seed['init'], seed['train'])
            assert guide.splitlines().count(row) == 1, row
            records += 1
    assert records == identity_count
    for fam in ('exp_054_uncapped_five_seeds', 'exp_055_uncapped_four_seeds'):
        parent = 'experiments/' + fam + '/'
        protocol = load(parent + 'protocol.json')
        # Inspect only the top-level seed-table assignment; importing the
        # scientific experiment is unnecessary.
        driver = ast.parse(files[parent + 'driver.py'].decode())
        assignments = [n for n in driver.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'SEEDS' for t in n.targets)]
        assert len(assignments) == 1
        assignment = ast.unparse(assignments[0].value)
        assert assignment == "json.loads((HERE / 'protocol.json').read_text())['seeds']", fam
        allowed = set(protocol['seeds'])
        for name in ('driver.py', 'verify_results.py'):
            tree = ast.parse(files[parent + name].decode())
            literals = {n.value for n in ast.walk(tree)
                        if isinstance(n, ast.Constant) and isinstance(n.value, str)
                        and re.fullmatch(r's\d+', n.value)}
            assert literals <= allowed, 'Unreported seed ID in ' + parent + name
        for name, expected in load(parent + 'code_manifest.json').items():
            assert digest(files[parent + name]) == expected, parent + name
        frozen = load(parent + 'frozen_runtime_config_sha256.json')
        for name, expected in frozen.items():
            assert digest(files['experiments/exp_040_scale_generalization_v2/' + name]) == expected
    provenance = load('reproducibility/source_provenance.json')
    source_rows = provenance['python_sources']
    for row in source_rows:
        assert digest(files[row['path']]) == row['sha256'], row['path']
    for name, data in files.items():
        text = data.decode('utf-8')
        for match in re.finditer(r'(?<![A-Za-z0-9])s(\d+)(?![A-Za-z0-9])', text):
            assert int(match[1]) <= 25, 'Unreported model ID in ' + name
    return records, len(source_rows), trajectories


def check_selector(files):
    # Extract only the pure decision function; never import JAX or the experiment.
    name = 'experiments/exp_040_scale_generalization_v2/path_set.py'
    tree = ast.parse(files[name].decode())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'choose_prefix')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), name, 'exec'), namespace)
    choose = namespace['choose_prefix']
    cases = [([.1, .5, .8], 2, 'smallest_prefix_reaching_target'),
             ([.6, .7], 1, 'smallest_prefix_reaching_target'),
             ([.2, .4, .4], 2, 'target_not_reached_max_coverage_prefix'),
             ([None, float('nan')], 1, 'no_finite_coverage_default_top1'),
             ([.1] * 23 + [.51], 24, 'smallest_prefix_reaching_target'),
             ([None, .49], 2, 'target_not_reached_max_coverage_prefix')]
    for coverages, selected, reason in cases:
        assert choose(coverages, .5) == (selected, reason)
    policy = json.loads(files['reproducibility/path_selection_policy.json'])
    assert policy['candidate_count_limit'] is None and policy['selected_path_count_limit'] is None
    assert policy['coverage']['target'] == .5 and policy['confirmation_used_for_selection'] is False
    assert json.loads(files['experiments/exp_040_scale_generalization_v2/settings.json'])['path_set']['k_max'] is None
    return len(cases) + check_discovery(files) + check_selection_evidence(files, choose)


def check_discovery(files):
    """Execute the actual exp040 discovery orchestration with synthetic records.

    No model, scientific import, disk writes or saved-result analysis. A target
    first reached at prefix five catches the old loop even if choose_prefix
    itself accepts arbitrarily long lists.
    """
    parent = 'experiments/exp_040_scale_generalization_v2/'

    def functions(name, names, namespace):
        tree = ast.parse(files[name].decode())
        body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert len(body) == len(names)
        exec(compile(ast.Module(body=body, type_ignores=[]), name, 'exec'), namespace)

    pure = dict(Counter=Counter, Integral=Integral, HEADS=8)
    functions(parent + 'path_set.py', {'rank_paths', 'choose_prefix', 'describe', 'random_control_set'}, pure)
    paths = SimpleNamespace(**{name: pure[name] for name in
                            ('rank_paths', 'choose_prefix', 'describe', 'random_control_set')})
    ranking = [head + 8 * layer for head in range(8) for layer in range(3)]
    scores = {p: float(len(ranking) - i) for i, p in enumerate(ranking)}
    scenarios = [([.1] * 4 + [.6] * 20, 5),
                 ([.6] * 24, 1),
                 ([.1] * 4 + [.4, .4] + [.2] * 18, 5),
                 ([None] * 24, 1)]
    for coverages, expected_k in scenarios:
        saved, evaluated = {}, []

        class Archive:
            def __init__(self, path, **kwargs):
                self.record = saved[str(path)]
                self.files = list(self.record)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def __getitem__(self, key):
                return self.record[key]

        def evaluate(model, data, ids, selected, **kwargs):
            evaluated.append(tuple(selected))
            return dict(paths=tuple(selected)), True

        runtime = SimpleNamespace(
            mechanism_evaluate=evaluate, structure_passed=bool,
            gain=lambda rec, task, kind: SimpleNamespace(mean=lambda: scores[rec['paths'][0]]),
            coverage=lambda rec: (coverages[len(rec['paths']) - 1], {}))
        namespace = dict(
            PS=paths, R=runtime, np=SimpleNamespace(load=Archive, mean=lambda xs: sum(xs) / len(xs)),
            SETTINGS=dict(path_set=dict(k_max=None, coverage_target=.5),
                          protection_tasks=['icl', 'ciwl'], random_path_seed=123),
            M_candidates=lambda model: range(24), load_dataset=lambda name: ({}, {}),
            save_npz=lambda path, rec: saved.__setitem__(str(path), rec),
            sha=lambda path: 'synthetic', heartbeat=lambda **kwargs: None, DATA=Path('synthetic'))
        functions(parent + 'worker.py', {'discover'}, namespace)
        result = namespace['discover'](dict(id='s13', init=13, train=10), object(), Path('synthetic'))
        assert result['ranking'] == ranking
        assert result['k'] == expected_k and result['selected'] == sorted(ranking[:expected_k])
        assert result['candidate_count'] == 24 and result['artificial_cap'] is None
        assert result['prefixes'] == [sorted(ranking[:k]) for k in range(1, 25)]
        assert result['prefix_coverage'] == coverages
        assert len(evaluated) == 24 + 23  # prefix one reuses its singleton
    # The exp047 utility must also enumerate its actual candidate count.
    tree = ast.parse(files['experiments/exp_047_cross_layer_label_k/driver.py'].decode())
    discovery = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'discover_stage')
    loops = [ast.unparse(n.iter) for n in ast.walk(discovery)
             if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'k']
    assert loops == ['range(1, len(ranking) + 1)']
    return len(scenarios)


def check_selection_evidence(files, choose):
    record = json.loads(files['reproducibility/uncapped_selection_validation.json'])
    selected = json.loads(files['reproducibility/selected_pathways.json'])
    expected = {(r['model'], r['seed_id']): r['selected_path_ids']
                for r in selected['trajectories'] if r['experiment_group'] == 'two_layer_scale'}
    actual = {}
    for row in record['two_layer_scale']:
        key = (row['model'], row['seed_id'])
        assert key not in actual
        actual[key] = row['selected_path_ids']
        k, reason = choose(row['saved_discovery_prefix_coverage'], .5)
        assert reason == 'smallest_prefix_reaching_target' and k == len(actual[key])
    assert actual == expected and len(actual) == 10
    assert record['raw_discovery_recomputed'] is False
    assert len(record['unverified_trajectories']) == 7
    return len(actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--git-index', action='store_true', help='Inspect exact staged blobs before commit.')
    parser.add_argument('--update-manifest', action='store_true', help='Rebuild hashes from the explicit release allowlist.')
    args = parser.parse_args()
    if args.git_index and args.update_manifest:
        parser.error('Do not update the manifest while reading the Git index.')
    files = index_files() if args.git_index else disk_files()
    policy = json.loads(files[POLICY])
    errors = []
    allowed = set(policy['allowed_paths'])
    if len(allowed) != len(policy['allowed_paths']):
        errors.append('Duplicate allowlist path')
    for path in sorted(set(files) - allowed - {'MANIFEST.json'}):
        errors.append('Outside release allowlist: ' + path)
    for path in sorted(allowed - set(files)):
        errors.append('Missing allowed file: ' + path)
    check_paths(files, policy, errors)
    seeds = sources = cases = trajectories = 0
    try:
        seeds, sources, trajectories = check_seeds(files)
        cases = check_selector(files)
        binding = json.loads(files['reproducibility/manuscript_binding.json'])
        inventory = json.loads(files['reproducibility/paper_inventory.json'])
        assert inventory['manuscript_sha256'] == binding['source_sha256']
    except (AssertionError, KeyError, StopIteration, ValueError) as exc:
        errors.append('Source/configuration check failed: ' + str(exc))
    if args.update_manifest and not errors:
        manifest = dict(schema='icl_paper_code_only_release_v1',
                        manuscript_sha256=binding['source_sha256'],
                        manifest_self_excluded=True,
                        scope='Paper code, necessary configuration and cited pathway/statistical metadata; no raw scientific data or full experimental outputs.',
                        files=[dict(path=name, bytes=len(files[name]), sha256=digest(files[name]))
                               for name in sorted(allowed)])
        data = (json.dumps(manifest, indent=2) + '\n').encode()
        (ROOT / 'MANIFEST.json').write_bytes(data)
        files['MANIFEST.json'] = data
        if sum(map(len, files.values())) > policy['max_total_bytes']:
            errors.append('Manifest pushes release over the total budget')
    if 'MANIFEST.json' in files:
        manifest = json.loads(files['MANIFEST.json'])
        entries = manifest['files']
        if len({r['path'] for r in entries}) != len(entries) or {r['path'] for r in entries} != allowed:
            errors.append('Manifest differs from explicit release allowlist')
        for row in entries:
            data = files.get(row['path'], b'')
            if len(data) != row['bytes'] or digest(data) != row['sha256']:
                errors.append('Manifest size/hash mismatch: ' + row['path'])
    else:
        errors.append('Missing MANIFEST.json')
    report = dict(passed=not errors, snapshot='git_index' if args.git_index else 'disk',
                  files=len(files), total_bytes=sum(map(len, files.values())),
                  limit_bytes=policy['max_total_bytes'],
                  python_files_parsed=sum(name.endswith('.py') for name in files),
                  paper_python_sources_checked=sources, seed_pairs_checked=seeds,
                  seed_architecture_records_checked=trajectories,
                  selector_cases_checked=cases, scientific_modules_imported=False,
                  experiment_data_included=False, model_or_training_executed=False, errors=errors)
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
