"""Check Appendix A against the included 201-point saved baseline curve."""
from pathlib import Path
import csv
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'experiments/exp_002_main_reproduction/final_00020000000/learning_curves.tsv'

def main():
    with PATH.open(newline='') as f:
        rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(f,delimiter='\t')]
    peak=max(rows,key=lambda r:r['icl_in_context_acc'])
    end=max(rows,key=lambda r:r['training_sequences'])
    checks={
        'saved_points_201':len(rows)==201,
        'peak_stage_4_4M':peak['training_sequences']==4_400_000,
        'peak_two_candidate_ICL_76_98_percent':abs(peak['icl_in_context_acc']-.7698)<1e-12,
        'terminal_stage_20M':end['training_sequences']==20_000_000,
        'terminal_two_candidate_ICL_54_08_percent':abs(end['icl_in_context_acc']-.5408)<1e-12,
        'terminal_two_candidate_CIWL_96_74_percent':abs(end['ciwl_in_context_acc']-.9674)<1e-12,
    }
    print(json.dumps({'passed':all(checks.values()),'checks':checks,
        'source_sha256':hashlib.sha256(PATH.read_bytes()).hexdigest(),
        'peak':peak,'terminal':end,
        'scope':'Appendix A saved two-candidate learning curve; no model evaluation and no full-label protection comparison.'},indent=2))
    if not all(checks.values()):raise SystemExit(1)

if __name__=='__main__':main()
