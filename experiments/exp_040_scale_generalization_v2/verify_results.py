"""Independent readback of exp040 outputs. Does not import analyze.py, timing.py,
path_set.py or any JAX code; the rules are re-implemented from protocol.md.

Checks: timing locks from saved monitor curves; discovery ranking, prefix
coverage, selected set and random set from saved discovery rows; receipt and
output SHA-256 bindings; fork identity of every arm; window means, primary
contrasts, seed-level t statistics and Holm adjustment against analysis.json.
"""
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import special

WORK = Path(__file__).resolve().parent
S = json.loads((WORK / 'settings.json').read_text())
OUT = Path(os.environ.get('ICL040_OUTPUTS', str(WORK / 'outputs'))).resolve()
H = 100000
checks, failures = [0], []


def ok(condition, message):
    checks[0] += 1
    if not condition:
        failures.append(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def lock_rule(curve):
    cfg = S['timing']
    for horizon in (cfg['first_horizon'], cfg['extended_horizon']):
        pts = {t: v for t, v in curve.items() if t <= horizon}
        sm = {t: (pts[t - H] + pts[t] + pts[t + H]) / 3 for t in pts if t - H in pts and t + H in pts}
        if not sm:
            return None
        peak = max(sm.values())
        tp = min(t for t in sm if sm[t] == peak)
        near = [t for t in sorted(sm) if all(sm.get(t + k * H, -1) >= peak - cfg['near_peak_tolerance'] for k in range(3))]
        dec = [t for t in sorted(sm) if t > tp and sm[t] <= peak - cfg['decline_anchor']]
        if near and dec:
            start = max(near[0], dec[0] - cfg['continuation_sequences'])
            return dict(start=start, end=start + cfg['continuation_sequences'], peak=peak, s0=sm[start])
        if not near:
            return None
    return None


def random_set(init, train, selected):
    selected = tuple(sorted(selected))
    material = (f"exp040/random-set/v1|init={init}|train={train}|random_path_seed={S['random_path_seed']}"
                f"|selected={','.join(map(str, selected))}").encode('ascii')
    rng = random.Random(int.from_bytes(hashlib.sha256(material).digest(), 'big'))
    out = []
    for source, count in sorted(Counter(p // 8 for p in selected).items()):
        out += rng.sample([source * 8 + h for h in range(8) if source * 8 + h not in selected], count)
    return sorted(out)


def gains(path):
    with np.load(path, allow_pickle=False) as r:
        res = {}
        for t in S['protection_tasks']:
            n = r[t + '/native/donor_ce'].astype(np.float64)
            res[t] = ((n - r[t + '/path/donor_ce'].astype(np.float64)).mean(), (n - r[t + '/donor/donor_ce'].astype(np.float64)).mean())
        return res


def t_sf(t, df):
    return 0.5 * special.betainc(df / 2, 0.5, df / (df + t * t)) if t > 0 else 1 - 0.5 * special.betainc(df / 2, 0.5, df / (df + t * t))


def main():
    analysis = load(OUT / 'analysis.json')
    for arch in S['architecture_order']:
        A = analysis['architectures'][arch]
        for spec in S['training_seeds']:
            sid, root = spec['id'], OUT / arch / spec['id']
            if not (root / 'timing.json').exists():
                failures.append(f'{arch}/{sid}: missing timing lock')
                continue
            lock = load(root / 'timing.json')
            curve = {p['stage']: p['monitor']['icl']['acc'] for p in load(root / 'normal' / 'curve.json')['points']}
            ok(sha(root / 'normal' / 'curve.json') == lock['curve_sha256'], f'{arch}/{sid}: curve changed after lock')
            rule = lock_rule(curve)
            if lock['applicable']:
                ok(rule is not None and rule['start'] == lock['start_stage'] and rule['end'] == lock['end_stage'],
                   f'{arch}/{sid}: timing lock not reproduced')
                win = [curve[t] for t in range(lock['end_stage'] - S['timing']['window_sequences'], lock['end_stage'] + 1, H)]
                ok(rule['peak'] >= S['timing']['min_peak'] and rule['s0'] - sum(win) / len(win) >= S['timing']['min_window_decline'],
                   f'{arch}/{sid}: applicability not reproduced')
            else:
                ok(rule is None or rule['peak'] < S['timing']['min_peak'] or rule['end'] > S['timing']['max_normal_end']
                   or rule['s0'] - np.mean([curve.get(t, np.nan) for t in range(rule['end'] - S['timing']['window_sequences'], rule['end'] + 1, H)]) < S['timing']['min_window_decline'],
                   f'{arch}/{sid}: non-applicability not reproduced')
                continue
            sel = load(root / 'path_selection.json')
            scores = {}
            for p in sel['candidate_domain']:
                g = gains(root / 'discovery' / f'single{p}.npz')
                scores[p] = float(np.mean([g[t][0] for t in S['protection_tasks']]))
                ok(abs(scores[p] - sel['discovery_scores'][str(p)]) < 1e-9, f'{arch}/{sid}: discovery score {p}')
            ranking = sorted(scores, key=lambda p: (-scores[p], p))
            ok(ranking == sel['ranking'], f'{arch}/{sid}: ranking')
            covs = []
            for k in range(1, len(sel['prefixes']) + 1):
                f = root / 'discovery' / (f'single{ranking[0]}.npz' if k == 1 else f'prefix{k}.npz')
                g = gains(f)
                c = [g[t][0] / g[t][1] if g[t][1] > 0 else None for t in S['protection_tasks']]
                covs.append(None if None in c else min(c))
                ok(sorted(ranking[:k]) == sel['prefixes'][k - 1], f'{arch}/{sid}: prefix {k}')
            target = S['path_set']['coverage_target']
            reached = [k + 1 for k, c in enumerate(covs) if c is not None and c >= target]
            finite = [(k + 1, c) for k, c in enumerate(covs) if c is not None]
            k_star = min(reached) if reached else (min(k for k, c in finite if c == max(c for _, c in finite)) if finite else 1)
            ok(k_star == sel['k'] and sorted(ranking[:k_star]) == sel['selected'], f'{arch}/{sid}: selected set')
            ok(random_set(spec['init'], spec['train'], sel['selected']) == sel['random_set'], f'{arch}/{sid}: random set')
            mech = load(root / 'mechanism_complete.json')
            ok(mech['selected'] == sel['selected'] and mech['random_set'] == sel['random_set'], f'{arch}/{sid}: mechanism receipt')
            ok(mech['a_fork_exactness']['passed'], f'{arch}/{sid}: A fork exactness')
            for arm in S['arms']:
                c = root / arm / 'complete.json'
                if not c.exists():
                    failures.append(f'{arch}/{sid}/{arm}: arm not complete')
                    continue
                r = load(c)
                ok(r['start_checkpoint_sha256'] == lock['start_checkpoint_sha256'], f'{arch}/{sid}/{arm}: fork identity')
                ok(r['end_stage'] == lock['end_stage'], f'{arch}/{sid}/{arm}: end stage')
                for rel, digest in r['outputs_file_sha256'].items():
                    ok(sha(OUT / rel) == digest, f'{arch}/{sid}/{arm}: output changed {rel}')
            # window means re-derived with a different reduction order
            for arm in ('A',) + tuple(S['arms']):
                files = [root / arm / 'confirm' / f'{arm}_{t:011d}.npz' for t in lock['window_stages']]
                if not all(f.exists() for f in files):
                    continue
                total = None
                for f in files:
                    with np.load(f, allow_pickle=False) as r:
                        v = r['icl/acc'].astype(np.float64)
                    total = v if total is None else total + v
                mean = float((total / len(files)).sum() / total.size)
                ok(abs(mean - A['seeds'][sid]['means'][arm]['icl']['acc']['mean']) < 1e-9, f'{arch}/{sid}/{arm}: ICL window mean')
    tests = analysis['primary']
    raw = []
    for t in tests:
        arch, (a, b, task, metric) = t['architecture'], S['analysis']['primary_hypotheses'][t['hypothesis']]
        seeds = analysis['architectures'][arch]['seeds']
        vals = [s['means'][a][task][metric]['mean'] - s['means'][b][task][metric]['mean']
                for s in seeds.values() if 'means' in s and a in s['means'] and b in s['means']]
        n = len(vals)
        if n >= 2:
            m = sum(vals) / n
            sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
            p = float(t_sf(m / (sd / math.sqrt(n)), n - 1)) if sd > 0 else (0.0 if m > 0 else 1.0)
            ok(abs(m - t['mean']) < 1e-9 and abs(p - t['p_raw']) < 1e-6, f'primary {arch} {t["hypothesis"]} recomputation')
        else:
            p = 1.0
        raw.append(p)
    order = sorted(range(len(raw)), key=lambda k: raw[k])
    running, adj = 0.0, [1.0] * len(raw)
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(raw) - rank) * raw[k]))
        adj[k] = running
    for t, a in zip(tests, adj):
        ok(abs(a - t['p_holm']) < 1e-6, f'Holm {t["architecture"]} {t["hypothesis"]}')
    result = dict(passed=not failures, checks=checks[0], failures=failures,
                  analysis_sha256=sha(OUT / 'analysis.json'), code_manifest_sha256=sha(WORK / 'code_manifest.json'))
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=result['passed'], checks=checks[0], failures=len(failures)), indent=1))
    return result


if __name__ == '__main__':
    main()
