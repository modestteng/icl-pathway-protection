"""Export already committed evaluator rows from the same baseline's two logs.

The cutoff was confirmed by a prior successful status read. Disable reader
locking so this optional export cannot block the training writer. Never read
the active tail: only the immutable, previously committed prefix is used.
"""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--cutoff',type=int,required=True)
a=p.parse_args()
root=Path(__file__).resolve().parent
runs=[root/'runs/main',root/'runs/main_resume_00002950016']
configs=[json.loads((run/'config.json').read_text()) for run in runs]
for name in ['depth','d_model','num_heads','class_split','exemplar_split',
             'data_file','train_bs','lr','optimizer','init_seed','train_seed',
             'eval_seed','load_eval_data']:
    assert configs[0][name]==configs[1][name],name
keys=['train_eval','icl','iwl_copy_avail','flip_icl','pure_iwl']
parts=[]
for run in runs:
    with h5py.File(run/'log.h5','r',locking=False) as f:
        x=np.asarray(f['eval_iter'])
        count=int(np.searchsorted(x,a.cutoff,side='right'))
        x=x[:count]
        ys=[]
        for key in keys:
            metric='acc' if key=='pure_iwl' else 'in_context_acc'
            values=np.asarray(f[key+'/'+metric][:count])
            assert values.shape==(count,5000)
            assert np.isfinite(values).all()
            assert np.all((values>=0)&(values<=1))
            ys.append(values.mean(axis=1))
        parts.append(np.column_stack([x,*ys]))
data=np.concatenate(parts)
assert np.all(np.diff(data[:,0])>0)
assert int(data[-1,0])==a.cutoff
assert len(data)==a.cutoff//100000+1
out=root/f'progress_{a.cutoff:011d}'
out.mkdir(exist_ok=True)
closed=False
if (root/'close_at_20m.json').exists() and (root/'final_checkpoint_audit.json').exists():
    stop=json.loads((root/'close_at_20m.json').read_text())
    audit=json.loads((root/'final_checkpoint_audit.json').read_text())
    closed=(stop['state']=='training_stopped_at_saved_boundary'
            and stop['target_sequences']==a.cutoff==audit['iter'] and audit['audit']=='passed')
np.savetxt(out/'learning_curves.tsv',data,delimiter='\t',
           header='training_sequences\ttrain_in_context_acc\ticl_in_context_acc\tciwl_in_context_acc\tflip_in_context_acc\tpure_iwl_acc',comments='')
colors=['#777777','#2463ac','#dc7c21','#299d8f']
names=['Training distribution','ICL','CIWL','Flip']
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for i,(name,color) in enumerate(zip(names,colors),1):
    axes[0].plot(data[:,0]/1e6,data[:,i],label=name,color=color,linewidth=2)
axes[0].axhline(.5,color='#999999',linestyle=':',label='Chance (2 labels)')
axes[0].set(xlabel='Training sequences (millions)',ylabel='In-context accuracy',ylim=(0,1))
axes[0].legend(loc='best')
axes[1].plot(data[:,0]/1e6,data[:,5],label='Pure IWL',color='#8b65af',linewidth=2)
axes[1].axhline(1/12800,color='#999999',linestyle=':',label='Chance (12800 labels)')
axes[1].set(xlabel='Training sequences (millions)',ylabel='Full-label accuracy',ylim=(0,.001))
axes[1].legend(loc='best')
phase='at frozen endpoint' if closed else 'in progress'
fig.suptitle(f'Main baseline {phase}: through {a.cutoff/1e6:g}M sequences\ninit seed 5 / train seed 2 / eval seed 1',fontsize=12)
fig.savefig(out/'learning_curves.png',dpi=180)
fig.savefig(out/'learning_curves.pdf')
report=dict(status=('completed user-approved integer endpoint; original 64M horizon truncated'
                    if closed else 'interim observed behavior; full 64M run still active'),
            cutoff_sequences=a.cutoff,evaluation_points=len(data),
            log_segments=[str(run/'log.h5') for run in runs],
            metrics={key:dict(initial=float(data[0,i]),latest=float(data[-1,i]),
                              maximum=float(data[:,i].max()),
                              maximum_at_sequences=int(data[data[:,i].argmax(),0]))
                     for i,key in enumerate(keys,1)})
(out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
