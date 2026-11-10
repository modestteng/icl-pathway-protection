"""One seed of the author's main experiment; no sweep or intervention.

The optional paper-window horizon retains the original training prefix and
saves/evaluates the exact 20M boundary. All model/data/optimizer parameters are
the author's main settings. Features must first pass the separately recorded
data-order approval and provenance audit.
"""
import argparse
import json
from pathlib import Path
import sys

p=argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--stage', choices=['eval','train'], required=True)
p.add_argument('--paper-window', action='store_true',
               help='Stop after saving/evaluating 20M sequences, the Figure 1b window')
p.add_argument('--resume-checkpoint', type=Path)
args=p.parse_args()
sys.path.insert(0,str(args.source.resolve()))

import jax
import main
import main_utils
import opto

assert jax.default_backend() == 'gpu', jax.devices()
args.root.mkdir(parents=True,exist_ok=True)
features=args.root/'features'
runs=args.root/'runs'
parser=main_utils.create_parser()
opto.add_args_to_parser(parser)
common=['--raw_name','--base_folder',str(runs),'--suppress_output',
        '--data_type','file','--pos_embedding_type','ape',
        '--depth','2','--d_model','64','--num_heads','8',
        '--init_seed','5','--train_seed','2','--eval_seed','1',
        '--mixing_coeffs','1.0','--pt_burstiness','1',
        '--train_context_len','2','--fs_relabel','0',
        '--lr','0.00001','--optimizer','adam']
if args.stage == 'eval':
    argv=common+[
        '--run','default_eval','--data_file',str(features/'omniglot_features_norotate.h5'),
        '--class_split','1600','0','23','--exemplar_split','5','0','15',
        '--train_iters','101','--eval_every','50','--eval_iters','5000',
        '--pe_names','train_eval','iwl_copy_avail','flip_icl','icl','pure_iwl',
        '--pe_classes',*(['train']*5),'--pe_exemplars',*(['train']*5),
        '--pe_burstiness','1','0','1','1','0',
        '--pe_fs_relabel_scheme','None','None','flip','01','None',
        '--pe_no_support','0','1','0','0','1',
        '--pe_unique_rest','0','1','0','0','1',
        '--pe_assign_query_label_random','0','1','0','0','0',
        '--save_eval_data','eval_data.h5']
else:
    horizon=20000032 if args.paper_window else 64000000
    run_name='main' if args.resume_checkpoint is None else 'main_resume_'+args.resume_checkpoint.stem
    argv=common+[
        '--run',run_name,'--data_file',str(features/'omniglot_features_reordered.h5'),
        '--class_split','12800','0','184','--exemplar_split','20','0','0',
        '--train_iters',str(horizon),'--eval_every','100000','--ckpt_every','50000',
        '--load_eval_data',str(runs/'default_eval'/'eval_data.h5')]
    if args.resume_checkpoint:
        argv+=['--load_from_ckpt',str(args.resume_checkpoint),
               '--load_from_ckpt_cfg',str(runs/'main'/'config.json')]
run_folder=runs/('default_eval' if args.stage=='eval' else run_name)
if (run_folder/'log.h5').exists():
    raise FileExistsError(f'{run_folder}; do not append an independent run to an existing log')
run_folder.mkdir(parents=True,exist_ok=True)
(run_folder/'argv.json').write_text(json.dumps(argv,indent=2)+'\n')
opts=parser.parse_args(argv)
print('GPU',jax.devices(), 'stage',args.stage,flush=True)
main.run_with_opts(opts)
