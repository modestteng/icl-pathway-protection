"""Standard-library seal and prerequisite identity check; no model execution."""
from pathlib import Path
import hashlib
import json

WORK = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate():
    manifest = json.loads((WORK / 'code_manifest.json').read_text())
    for name, expected in manifest.items():
        assert digest(WORK / name) == expected, ('sealed_source', name)
    s = json.loads((WORK / 'settings.json').read_text())
    assert s['arms'] == ['REPLAY', 'L2']
    assert s['training_seeds'] == [{'id': 's5', 'init': 5, 'train': 2}]
    assert (s['start_stage'], s['end_stage'], s['train_bs']) == (4400000, 10000000, 32)
    assert s['replay_coefficient'] == 2.0 and s['replay_loss_normalizer'] == 3.0
    assert s['l2_coefficient'] == 1e-4
    assert s['l2_form'] == 'sum_of_squared_all_trainable_array_entries_without_half'
    assert s['protection_tasks'] == ['icl', 'flip_icl']
    assert s['data_sets']['protection']['n'] == 4096 and s['protection_bs_per_task'] == 32
    assert s['confirm_stages'] == list(range(9000000, 10000001, 100000))
    refs = json.loads((WORK / 'references_manifest.json').read_text())
    for path, expected in refs['files'].items():
        assert digest(path) == expected, ('reference_sha', path)
    assert len(refs['raw_records']) == 33
    for row in refs['raw_records']:
        verifier = json.loads(Path(row['verifier_path']).read_text())
        assert verifier['passed'] is True and not verifier['failures']
        assert verifier['verified_file_sha256'][str(Path(row['path']).resolve())] == row['sha256']
    for filename, expected in s['preserved_data_sha256'].items():
        assert digest(WORK / 'outputs/data' / filename) == expected, filename
    print(json.dumps({'passed': True, 'sealed_files': len(manifest),
                      'reference_raw_files': len(refs['raw_records'])}), flush=True)


if __name__ == '__main__':
    validate()
