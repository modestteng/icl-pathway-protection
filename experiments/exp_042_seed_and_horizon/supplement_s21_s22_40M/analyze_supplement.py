"""Supplementary descriptive summaries; original registered tests remain immutable."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent
sys.path.insert(0, str(ORIGINAL))
from plan import sha, read_json, write_json
from statistics_core import summarize
import analyze as inherited


def describe(values):
    result = summarize(values, 5)
    return {k:v for k,v in result.items() if k not in
            ['p_greater', 'threshold', 'positive_seeds', 'inference']}


def main():
    frozen = read_json(ORIGINAL/'outputs/analysis.json')
    assert sha(ORIGINAL/'outputs/analysis.json') == read_json(HERE/'original_inventory.json')['original_analysis_sha256']
    rows = list(frozen['per_seed'])
    inherited.OUTPUTS = HERE/'outputs'
    for sid in ['s21','s22']:
        for arm in ['A','P25']:
            for budget in [10000000,20000000,30000000,40000000]:
                data, missing = inherited.load_window(sid, arm, budget)
                assert not missing, missing
                metrics = {k:float(v.mean()) for k,v in data.items() if k.endswith(('/acc','/ce'))}
                if budget == 10000000:
                    prior = next(r for r in rows if (r['seed'],r['arm'],r['budget']) == (sid,arm,budget))
                    assert all(abs(prior['metrics'][k]-v)<2e-12 for k,v in metrics.items())
                else:
                    rows.append(dict(seed=sid,arm=arm,budget=budget,metrics=metrics))
    lookup = {(r['seed'],r['arm'],r['budget']):r['metrics'] for r in rows}
    keys = list(rows[0]['metrics'])
    budgets = {}
    for budget in [10000000,20000000,30000000,40000000]:
        metrics = {}
        for key in keys:
            av = [lookup[s,'A',budget][key] for s in ['s18','s19','s20','s21','s22']]
            pv = [lookup[s,'P25',budget][key] for s in ['s18','s19','s20','s21','s22']]
            metrics[key] = dict(A=describe(av), P25=describe(pv),
                               difference=describe([p-a for a,p in zip(av,pv)]))
        budgets[str(budget)] = dict(seeds=['s18','s19','s20','s21','s22'],metrics=metrics)
    persistence = {key:describe([lookup[s,'P25',40000000][key]-lookup[s,'P25',10000000][key]
        for s in ['s18','s19','s20','s21','s22']]) for key in keys}
    result = dict(complete=True, code_manifest_sha256=sha(ORIGINAL/'code_manifest.json'),
        supplement_manifest_sha256=sha(HERE/'manifest.json'), per_seed=rows, budgets=budgets,
        P25_40M_minus_10M=persistence, analysis_scope='supplementary_descriptive_only_after_original_results',
        original_analysis_sha256=sha(ORIGINAL/'outputs/analysis.json'),
        original_primary_results_unchanged=frozen['primary'],
        interval_scope='training-seed variability conditional on the unchanged fixed evaluation pool')
    write_json(HERE/'outputs/analysis.json', result)
    lines = ['# exp042 s21/s22 40M 补充结果', '',
        '本补充在原结果可见后开展。原预定三种子检验保持原样；以下五种子汇总仅作补充描述。', '',
        '|种子|40M A 普通ICL / %|40M P25 普通ICL / %|差值 / pp|', '|---|---:|---:|---:|']
    for sid in ['s18','s19','s20','s21','s22']:
        a,p = [lookup[sid,arm,40000000]['icl/acc']*100 for arm in ['A','P25']]
        lines.append(f'|{sid}|{a:.4f}|{p:.4f}|{p-a:+.4f}|')
    lines.extend(['', '完整三任务准确率、交叉熵及四个训练量窗口见 analysis.json。'])
    (HERE/'outputs/report.md').write_text('\n'.join(lines)+'\n')
    print('SUPPLEMENT_ANALYSIS_COMPLETE')


if __name__ == '__main__':
    main()
