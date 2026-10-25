"""Render saved Section 5 data with the original numeric drawing functions.

Only the English path is called. Font registration, preview subprocesses and the
original __main__ block are skipped. No model, fitting or interval estimation.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    # Generated figures are not release inputs and must not overwrite the archive.
    if out==ROOT or ROOT in out.parents:
        parser.error('Use an output directory outside the extracted release package.')
    out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'figures/paper_figures/section5_20261022/draw_section5.py'
    tree=ast.parse(source.read_text(),filename=str(source))
    kept=[]
    skipped=[]
    for node in tree.body:
        if isinstance(node,ast.If) and '__name__' in ast.unparse(node.test):
            skipped.append('original __main__ block');continue
        if (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
            and isinstance(node.value.func,ast.Attribute)
            and node.value.func.attr=='registerFont'):
            skipped.append('platform-specific Chinese font registration');continue
        kept.append(node)
    tree.body=kept
    namespace={'__name__':'frozen_section5_source','__file__':str(source)}
    exec(compile(tree,str(source),'exec'),namespace)
    namespace['HERE']=out
    def finish(self):
        self.c.showPage();self.c.save()
        namespace['BOUNDS'].append({'file':self.path.name,'labels':self.bounds})
    namespace['Figure'].finish=finish
    for name in ['strength_plot','scale_plot','horizon_plot','external_plot']:
        namespace[name]('en')
    expected=json.loads((ROOT/'reproducibility/expected_figure_values.json').read_text())
    if namespace['VALUES']!=expected:
        raise AssertionError('Saved scientific plotting values differ from manuscript freeze.')
    generated=sorted(out.glob('fig_s5_*_en.pdf'))
    if len(generated)!=4:raise AssertionError('Expected four plots.')
    report={'passed':True,'all_numeric_plotting_inputs_equal_frozen_values':True,
            'numeric_drawing_functions':'original unmodified source definitions',
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'included_summary_sha256':namespace['SOURCES'],
            'skipped':skipped+['PDF preview subprocess'],
            'model_evaluation':False,'statistical_intervals_recomputed':False,
            'outputs':[p.name for p in generated]}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
