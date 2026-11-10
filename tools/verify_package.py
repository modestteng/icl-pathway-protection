"""Audit the exact code-only release, including the staged Git snapshot."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

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
    return len(cases)


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
                        scope='Only paper code and necessary configuration; no scientific data or results.',
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
