"""Plot author-defined metrics without smoothing or changing evaluators."""
import argparse
from pathlib import Path
import json
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('log',type=Path)
p.add_argument('--prefix-log',type=Path,
               help='Earlier segment of the same checkpoint-resumed run; requires non-overlap')
p.add_argument('--max-sequences',type=int,help='Explicit frozen checkpoint boundary')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
with h5py.File(a.log,'r') as f:
    x=np.array(f['eval_iter'])
    assert np.all(np.diff(x)>0), 'Inspect duplicate/non-monotone iterations before plotting'
    curves={k:np.array(f[f'{k}/in_context_acc']).mean(axis=1)
            for k in ['train_eval','icl','iwl_copy_avail','flip_icl']}
    pure=np.array(f['pure_iwl/acc']).mean(axis=1)
if a.prefix_log:
    with h5py.File(a.prefix_log,'r') as f:
        before=np.array(f['eval_iter'])
        assert np.all(np.diff(before)>0)
        assert before[-1]<x[0], 'Inspect overlapping resume evaluations before combining'
        for k in curves:
            curves[k]=np.concatenate([np.array(f[f'{k}/in_context_acc']).mean(axis=1),curves[k]])
        pure=np.concatenate([np.array(f['pure_iwl/acc']).mean(axis=1),pure])
        x=np.concatenate([before,x])
if a.max_sequences is not None:
    keep=x<=a.max_sequences
    x=x[keep]
    curves={k:y[keep] for k,y in curves.items()}
    pure=pure[keep]
    assert int(x[-1])==a.max_sequences, 'Requested boundary evaluation is absent'
labels={'train_eval':'Training distribution','icl':'ICL',
        'iwl_copy_avail':'CIWL','flip_icl':'Flip'}
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for k,y in curves.items():axes[0].plot(x/1e6,y,label=labels[k])
axes[0].axhline(.5,color='gray',linestyle=':',label='Chance (2 labels)')
axes[0].set(xlabel='Training sequences (millions)',ylabel='In-context accuracy',ylim=(0,1))
axes[0].legend()
axes[1].plot(x/1e6,pure,label='Pure IWL')
axes[1].axhline(1/12800,color='gray',linestyle=':',label='Chance (12800 labels)')
axes[1].set(xlabel='Training sequences (millions)',ylabel='Full-label accuracy',ylim=(0,1))
axes[1].legend()
fig.savefig(a.output/'learning_curves.png',dpi=200)
fig.savefig(a.output/'learning_curves.pdf')
# Figure 1b uses the prefix below 20M sequences; show that same window from
# this single run in addition to the complete author 64M training history.
window=x<20000000
if np.any(window):
    prefix,ax=plt.subplots(figsize=(6,4),layout='constrained')
    for k,y in curves.items():ax.plot(x[window]/1e6,y[window],label=labels[k])
    ax.axhline(.5,color='gray',linestyle=':',label='Chance (2 labels)')
    ax.set(xlabel='Training sequences (millions)',ylabel='In-context accuracy',ylim=(0,1))
    ax.legend()
    prefix.savefig(a.output/'figure1b_window.png',dpi=200)
    prefix.savefig(a.output/'figure1b_window.pdf')
report={'status':'observed curves require scientific interpretation; no causal conclusion',
        'log_segments':[str(a.prefix_log),str(a.log)] if a.prefix_log else [str(a.log)],
        'evaluation_points':len(x),'last_training_sequences':int(x[-1]),
        'paper_window_points':int(window.sum()),
        'metrics':{k:{'initial':float(y[0]),'maximum':float(y.max()),
                      'maximum_at_sequences':int(x[np.argmax(y)]),'final':float(y[-1])}
                   for k,y in {**curves,'pure_iwl':pure}.items()}}
(a.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
np.savetxt(a.output/'learning_curves.tsv',
           np.column_stack([x,*curves.values(),pure]),delimiter='\t',
           header='training_sequences\ttrain_in_context_acc\ticl_in_context_acc\tciwl_in_context_acc\tflip_in_context_acc\tpure_iwl_acc',comments='')
print(json.dumps(report,indent=2))
