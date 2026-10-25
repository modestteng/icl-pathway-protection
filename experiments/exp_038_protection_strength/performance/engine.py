"""Queue unchanged author kernels; commit only after the full chunk is checked."""
import science as R
from common import *
import json,time,subprocess,threading

CHUNK=16


@eqx.filter_jit
def split_keys(data_key, model_key):
    dn,dk=jax.random.split(data_key)
    mn,mk=jax.random.split(model_key)
    return dn,dk,mn,mk


@eqx.filter_jit
def packed_status(model,state,metrics,kind):
    losses=(metrics['loss'],metrics.get('protection_loss',jnp.asarray([])))
    finite=R.finite_candidate(model,(state,losses)).astype(jnp.float32)
    if kind=='A':return jnp.stack((finite,jnp.float32(1),jnp.float32(0),jnp.float32(1),jnp.float32(0),jnp.float32(0)))
    d=metrics['projection']
    return jnp.stack((finite,d['success'].astype(jnp.float32),d['relative_constraint_violation'],d['update_norm_ratio'],d['correction_norm_ratio'],d['projected'].astype(jnp.float32)))


def cache_protection(data):
    count=SETTINGS['data_sets']['protection']['n']//SETTINGS['protection_bs_per_task']
    values=[R.protect_batch(data,i) for i in range(count)]
    sync(values)
    return values


def stage_on_gpu(data):return {task:tuple(jax.device_put(v) for v in xy) for task,xy in data.items()}


def queue_steps(ck,count,kind,rho,head,sample,optimizer,cache):
    # The accepted caller state is not mutated. All candidate states in this
    # transaction remain tentative until every finite/projection check passes.
    current=dict(ck,seeds=dict(ck['seeds']))
    pending=[];audit=None
    for number in range(count):
        old=current
        dn,dk,mn,mk=split_keys(old['seeds']['train_data_seed'],old['seeds']['train_model_seed'])
        batch=sample(dk)
        offset=(int(old['iter'])-SETTINGS['start_stage'])//BS
        if kind=='A':
            metrics,model,state=main.train_step(model=old['model'],fwd_fn=R.FWD,optimizer=optimizer,opt_state=old['opt_state'],microbs=BS,weight_decay=0.,x=batch['examples'],y=batch['labels'],key=mk)
        else:
            px,py=cache[offset%len(cache)]
            metrics,model,state,raw,gradients=R.modified_step(old['model'],old['opt_state'],batch['examples'],batch['labels'],mk,px,py,kind,rho,head,optimizer)
            if offset in SETTINGS['diagnostic_update_offsets']:
                assert count==1,'audit updates remain individual transaction boundaries'
                audit=(int(old['iter']),old['model'],model,raw,gradients,px,py)
        pending.append(packed_status(model,state,metrics,kind))
        keys=dict(old['seeds'],train_data_seed=dn,train_model_seed=mn)
        current=dict(old,model=model,opt_state=state,iter=int(old['iter'])+BS,seeds=keys)
    status=np.asarray(jax.device_get(jnp.stack(pending)))
    valid=(status[:,0]==1)&(status[:,1]==1)&np.isfinite(status).all(axis=1)&(status[:,2]<=SETTINGS['numeric_projection_relative_violation_max'])
    if not valid.all():
        raise RuntimeError('Uncommitted chunk rejected; first invalid offset='+str(int(np.flatnonzero(~valid)[0]))+' status='+repr(status.tolist()))
    return current,status,audit


def serial_steps(ck,count,kind,rho,head,sample,optimizer,protect):
    current=dict(ck,seeds=dict(ck['seeds']));statuses=[]
    for _ in range(count):
        dn,dk=jax.random.split(current['seeds']['train_data_seed']);mn,mk=jax.random.split(current['seeds']['train_model_seed'])
        batch=sample(dk);offset=(int(current['iter'])-SETTINGS['start_stage'])//BS
        if kind=='A':
            metrics,model,state=main.train_step(model=current['model'],fwd_fn=R.FWD,optimizer=optimizer,opt_state=current['opt_state'],microbs=BS,weight_decay=0.,x=batch['examples'],y=batch['labels'],key=mk)
        else:
            px,py=R.protect_batch(protect,offset)
            metrics,model,state,raw,grads=R.modified_step(current['model'],current['opt_state'],batch['examples'],batch['labels'],mk,px,py,kind,rho,head,optimizer)
        finite=bool(np.asarray(R.finite_candidate(model,(state,(metrics['loss'],metrics.get('protection_loss',jnp.asarray([])))))))
        if kind=='A':row=[float(finite),1.,0.,1.,0.,0.]
        else:
            d=metrics['projection'];row=[float(finite),float(np.asarray(d['success'])),float(np.asarray(d['relative_constraint_violation'])),float(np.asarray(d['update_norm_ratio'])),float(np.asarray(d['correction_norm_ratio'])),float(np.asarray(d['projected']))]
        assert row[0] and row[1] and row[2]<=SETTINGS['numeric_projection_relative_violation_max']
        statuses.append(row)
        current=dict(current,model=model,opt_state=state,iter=int(current['iter'])+BS,seeds=dict(current['seeds'],train_data_seed=dn,train_model_seed=mn))
    return current,np.asarray(statuses,np.float32)

