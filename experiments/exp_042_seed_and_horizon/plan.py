"""Frozen plan and file operations; standard library only, safe on the Mac."""
from pathlib import Path
import hashlib
import json
import os

WORK = Path(__file__).resolve().parent
SETTINGS = json.loads((WORK / 'settings.json').read_text())
OUTPUTS = Path(os.environ.get('ICL042_OUTPUTS', str(WORK / 'outputs'))).resolve()
ASSETS = Path(os.environ.get('ICL042_ASSETS', str(WORK / 'assets'))).resolve()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def code_identity():
    return sha(WORK / 'code_manifest.json')


def verify_code():
    for name, expected in read_json(WORK / 'code_manifest.json').items():
        if sha(WORK / name) != expected:
            raise RuntimeError('sealed code changed: ' + name)


def seed_spec(name):
    return next(s for s in SETTINGS['training_seeds'] if s['id'] == name)


def window(budget):
    return list(range(budget - SETTINGS['window_sequences'], budget + 1, SETTINGS['eval_every']))


def budgets(seed):
    return [10000000, 20000000, 30000000, 40000000] if seed in SETTINGS['long_seeds'] else [10000000]


def protection_offset(stage):
    if stage < SETTINGS['start_stage'] or stage % SETTINGS['train_bs']:
        raise ValueError('invalid protected stage')
    return (stage - SETTINGS['start_stage']) // SETTINGS['train_bs']


def validate_settings():
    s = SETTINGS
    ids = [v['id'] for v in s['training_seeds']]
    assert ids == ['s18', 's19', 's20', 's21', 's22']
    assert s['long_seeds'] == ids[:3]
    assert len({(v['init'], v['train']) for v in s['training_seeds']}) == 5
    assert s['arms'] == ['A', 'P25'] and s['primary_relaxation'] == .25
    assert s['start_stage'] == 4400000 and s['short_end_stage'] == 10000000
    assert s['long_end_stage'] == 40000000
    assert s['train_bs'] == s['train_microbs'] == 32
    assert s['reported_tasks'] == ['icl', 'iwl_copy_avail', 'train_eval']
    assert s['execution']['workers_per_gpu'] == 1
    assert all(len(window(b)) == 11 for b in [10000000, 20000000, 30000000, 40000000])


def graph():
    """Names are stable across restarts. No experimental code is imported."""
    validate_settings()
    jobs = []
    def add(name, kind, deps=(), **args):
        jobs.append(dict(name=name, kind=kind, deps=list(deps), **args))
    add('data', 'data')
    add('validate_A', 'validate', ['data'], arm='A')
    add('validate_P25', 'validate', ['data'], arm='P25')
    short = []
    for spec in SETTINGS['training_seeds']:
        sid = spec['id']
        prefix = sid + '_prefix'
        add(prefix, 'prefix', ['validate_A', 'validate_P25'], seed=sid)
        discover = []
        for head in range(8):
            name = f'{sid}_discover_{head}'
            add(name, 'discover', [prefix], seed=sid, head=head)
            discover.append(name)
        select = sid + '_select'
        add(select, 'select', discover, seed=sid)
        confirm = sid + '_confirm'
        add(confirm, 'confirm', [select], seed=sid)
        preflight = sid + '_preflight'
        add(preflight, 'preflight', [confirm], seed=sid)
        for arm in SETTINGS['arms']:
            name = f'{sid}_{arm}_10M'
            add(name, 'train', [preflight], seed=sid, arm=arm, budget=10000000)
            late = name + '_diagnostic'
            add(late, 'diagnostic', [name], seed=sid, arm=arm, budget=10000000)
            short.append(late)
    add('short_complete', 'short_complete', short)
    for sid in SETTINGS['long_seeds']:
        for arm in SETTINGS['arms']:
            name = f'{sid}_{arm}_40M'
            add(name, 'train', ['short_complete', f'{sid}_{arm}_10M'], seed=sid, arm=arm, budget=40000000)
            add(name + '_diagnostic', 'diagnostic', [name], seed=sid, arm=arm, budget=40000000)
    return jobs
