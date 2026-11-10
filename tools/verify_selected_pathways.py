"""Validate the published pathway metadata without importing scientific code."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path,
                        default=ROOT / 'reproducibility/selected_pathways.json',
                        help='Selected-pathway JSON file; defaults to the published record.')
    parser.add_argument('--depth-results', type=Path, help='Optional local depth-results JSON file.')
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    registry = json.loads((ROOT / 'reproducibility/manuscript_seed_registry.json').read_text())
    expected = {
        (group['id'], model, seed['id'], seed['init'], seed['train'])
        for group in registry['groups']
        if group['id'] in ('two_layer_scale', 'four_layer_depth')
        for model in group['architectures'] for seed in group['seeds']
    }
    actual = set()
    counts = {}
    for row in record['trajectories']:
        key = (row['experiment_group'], row['model'], row['seed_id'],
               row['init_seed'], row['train_seed'])
        assert key not in actual, ('duplicate trajectory', key)
        actual.add(key)
        layers = int(row['model'].split('L/')[0])
        n = (layers - 1) * record['indexing']['heads_per_layer']
        paths = row['selected_path_ids']
        assert row['candidate_count'] == n
        assert len(paths) == row['selected_pathway_count'] > 0
        assert paths == sorted(set(paths))
        assert all(isinstance(p, int) and 0 <= p < n for p in paths)
        counts[str(key)] = [sum(p // 8 == layer for p in paths)
                            for layer in range(layers - 1)]
        source = ROOT / row['source_file']
        if row['experiment_group'] == 'four_layer_depth' and source.exists():
            saved = json.loads(source.read_text())
            assert saved['seed'] == dict(id=row['seed_id'], init=row['init_seed'],
                                         train=row['train_seed'])
            assert sorted(saved['selected']) == paths
            assert saved['candidate_count'] == n
    assert actual == expected and len(actual) == 18
    depth_path = args.depth_results
    if depth_path is not None:
        depth = json.loads(depth_path.read_text())['seeds']
        for row in record['trajectories']:
            if row['experiment_group'] == 'four_layer_depth':
                original = depth[row['seed_id']]
                assert sorted(original['selected']) == row['selected_path_ids']
                assert original['candidate_count'] == row['candidate_count']
    print(json.dumps(dict(passed=True, trajectories=len(actual),
                          seed_identity_fields=['group', 'architecture', 'ID', 'init', 'train'],
                          source_layer_counts=counts), indent=2))


if __name__ == '__main__':
    main()
