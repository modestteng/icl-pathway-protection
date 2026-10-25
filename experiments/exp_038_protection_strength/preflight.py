"""Actual GPU numerical checks and one bounded C/P/D dispatch comparison."""
from common import *
import science as R
from performance import engine
from projection import project_tree, project_numpy

PROJECT=eqx.filter_jit(project_tree)


def preflight():
    receipt=OUT/'preflight.json'
    if receipt.exists():
        p=json.loads(receipt.read_text())
        assert p['passed'] and p['code_manifest_sha256']==code_identity()
        return p
    started=time.monotonic();seed=SETTINGS['training_seeds'][0]
    base=base_checkpoint(seed);_,optimizer=template(seed);sample=make_sampler(seed)
    _,dk=jax.random.split(base['seeds']['train_data_seed']);_,mk=jax.random.split(base['seeds']['train_model_seed'])
    batch=sync(sample(dk));assert exact(batch,sample.native(dk,sample.data))
    protect,_=load_dataset('protection');px,py=R.protect_batch(protect,0)
    author=sync(main.train_step(model=base['model'],fwd_fn=R.FWD,optimizer=optimizer,
        opt_state=base['opt_state'],microbs=BS,weight_decay=0.,
        x=batch['examples'],y=batch['labels'],key=mk))
    checks={};grid={}
    for kind,count in [('C',2),('P',1),('D',3)]:
        result=sync(R.modified_step(base['model'],base['opt_state'],batch['examples'],batch['labels'],
            mk,px,py,kind,1.,0,optimizer))
        checks[kind+'_constraint_count']=len(result[4]);assert len(result[4])==count
        checks[kind+'_rho1_adam_exact']=exact(result[2],author[2]);assert checks[kind+'_rho1_adam_exact']
        checks[kind+'_rho1_model_max_abs']=maxdiff(result[1],author[1])
        assert checks[kind+'_rho1_model_max_abs']<=SETTINGS['numeric_model_adam_max_abs']
        raw,grads=result[3],result[4]
        grid[kind]={}
        for rho in [.1,.25,.5,.75]:
            projected,diag=sync(PROJECT(raw,grads,rho))
            vector,_=project_numpy(R.flat(raw),np.stack([R.flat(g) for g in grads]),rho)
            diff=R.flat(projected).astype(np.float64)-vector
            absolute=float(np.max(np.abs(diff)))
            relative=float(np.linalg.norm(diff)/max(np.linalg.norm(R.flat(raw).astype(np.float64)),1e-30))
            violation=float(np.asarray(diag['relative_constraint_violation']))
            assert bool(np.asarray(diag['success']))
            assert absolute<=SETTINGS['numeric_projection_reference_max_abs']
            assert relative<=SETTINGS['numeric_projection_reference_max_relative']
            assert violation<=SETTINGS['numeric_projection_relative_violation_max']
            # Exercise the actual production graph at every planned coefficient.
            actual=sync(R.modified_step(base['model'],base['opt_state'],batch['examples'],batch['labels'],
                mk,px,py,kind,rho,0,optimizer))
            assert exact(actual[2],author[2]) and exact(actual[3],raw)
            model_error=maxdiff(actual[1],eqx.apply_updates(base['model'],projected))
            assert model_error<=SETTINGS['numeric_model_adam_max_abs']
            grid[kind][f'{int(rho*100):02d}']=dict(passed=True,rho=rho,numpy_max_abs=absolute,
                numpy_relative=relative,relative_violation=violation,
                jit_relative_violation=float(np.asarray(actual[0]['projection']['relative_constraint_violation'])),
                production_model_max_abs=model_error)
        checks['projection_'+kind+'_numpy_max_abs']=max(v['numpy_max_abs'] for v in grid[kind].values())
        checks['projection_'+kind+'_numpy_relative']=max(v['numpy_relative'] for v in grid[kind].values())
        checks['projection_'+kind+'_relative_violation']=max(v['relative_violation'] for v in grid[kind].values())
        checks['projection_'+kind+'_jit_relative_violation']=max(v['jit_relative_violation'] for v in grid[kind].values())
        print('PREFLIGHT_KIND_PASSED',kind,flush=True)
    # Reconstruct actual Optax updates at first count and with nonzero history.
    gradient=grads[0]
    initial=optimizer.init(eqx.filter(base['model'],eqx.is_array))
    zero=jax.tree_util.tree_map(jnp.zeros_like,gradient)
    for label,g,state in [('first_count',gradient,initial),('zero_gradient_nonzero_moments',zero,base['opt_state'])]:
        expected,newstate=sync(R.official_optimizer_update(g,state,optimizer))
        rebuilt=sync(R.adam_update_from_state(newstate))
        delta=R.flat(rebuilt).astype(np.float64)-R.flat(expected).astype(np.float64)
        absolute=float(np.max(np.abs(delta)))
        relative=float(np.linalg.norm(delta)/max(np.linalg.norm(R.flat(expected)),1e-30))
        assert absolute<=SETTINGS['numeric_projection_reference_max_abs']
        assert relative<=SETTINGS['numeric_projection_reference_max_relative']
        checks[label]=dict(max_abs=absolute,relative=relative)
    xx,yy=px.reshape(-1,3,512),py.reshape(-1,3)
    keys=jax.random.split(KEY,len(xx));_,gradient=R.functional_objective(base['model'],xx,yy,keys,0)
    norm=float(np.asarray(R.tree_norm(gradient)));assert norm>0 and np.isfinite(norm)
    direction=jax.tree_util.tree_map(lambda v:v/norm,gradient)
    step=SETTINGS['numeric_functional_grad_fd_step']
    plus=eqx.apply_updates(base['model'],jax.tree_util.tree_map(lambda v:step*v,direction))
    minus=eqx.apply_updates(base['model'],jax.tree_util.tree_map(lambda v:-step*v,direction))
    lp,_=R.functional_objective(plus,xx,yy,keys,0);lm,_=R.functional_objective(minus,xx,yy,keys,0)
    fd=float(np.asarray(lp-lm))/(2*step)
    assert abs(fd-norm)<=SETTINGS['numeric_functional_grad_absolute_tolerance']+SETTINGS['numeric_functional_grad_relative_tolerance']*norm
    checks['functional_derivative']=dict(autodiff=norm,finite_difference=fd,abs_error=abs(fd-norm))
    original={t:(x[:128],y[:128]) for t,(x,y) in R.original_data().items()}
    native=main.evaluate(model=base['model'],fwd_fn=R.FWD,key=KEY,
        eval_data={t:dict(examples=jnp.asarray(x),labels=jnp.asarray(y)) for t,(x,y) in original.items()},
        eval_batch_size=SETTINGS['eval_bs'])
    adapted=R.evaluate(base['model'],original)
    for task in original:
        ce=abs(float(np.asarray(native[task]['loss']).mean())-float(adapted[task+'/loss'].mean()))
        rows=int(np.count_nonzero(np.asarray(native[task]['acc'])!=adapted[task+'/acc']))
        assert ce<=SETTINGS['numeric_eval_mean_ce_max_abs'] and rows<=SETTINGS['numeric_eval_acc_max_rows']
        checks[task+'_eval_ce_max_mean']=ce;checks[task+'_eval_acc_rows']=rows
    p=provenance()
    result=dict(passed=True,checks=checks,grid_checks=grid,limits=SETTINGS,code_manifest_sha256=code_identity(),
        inputs=p['inputs'],source=p['source'],model_forward_and_gradient_gpu_verified=True,
        evaluation_parity_rows_per_task=128,compute_seconds=time.monotonic()-started)
    write_json(receipt,result);print('PREFLIGHT_PASSED',flush=True)
    return result


def queued(ck,count,kind,rho,sample,optimizer,cache,chunk):
    records=[]
    while count:
        offset=(int(ck['iter'])-SETTINGS['start_stage'])//BS
        size=min(chunk,count)
        if offset in SETTINGS['diagnostic_update_offsets']:size=1
        else:
            future=[v-offset for v in SETTINGS['diagnostic_update_offsets'] if v>offset]
            if future:size=min(size,min(future))
        ck,status,_=engine.queue_steps(ck,size,kind,rho,0,sample,optimizer,cache)
        records.extend(status);count-=size
    return ck,np.asarray(records,np.float32)


def validate_performance():
    receipt=OUT/'performance_validation.json'
    if receipt.exists():
        result=json.loads(receipt.read_text())
        assert result['passed'] and result['code_manifest_sha256']==code_identity()
        return result
    seed=SETTINGS['training_seeds'][0];base=base_checkpoint(seed)
    sample=make_sampler(seed);_,optimizer=template(seed);protect,_=load_dataset('protection')
    assert SETTINGS['data_sets']['protection']['n']%SETTINGS['protection_bs_per_task']==0
    cache=engine.cache_protection(protect)
    assert all(exact(v,R.protect_batch(protect,i)) for i,v in enumerate(cache))
    started=time.monotonic();metrics={};timings={};selected={};reason={}
    for kind in ['C','P','D']:
        # All paths use the same unchanged scientific kernel objects.
        engine.serial_steps(base,2,kind,.5,0,sample,optimizer,protect)
        queued(base,2,kind,.5,sample,optimizer,cache,16)
        t=time.monotonic()
        before,strict_status=engine.serial_steps(base,64,kind,.5,0,sample,optimizer,protect)
        sync(before);strict_seconds=time.monotonic()-t
        t=time.monotonic()
        after,chunk_status=queued(base,64,kind,.5,sample,optimizer,cache,16)
        sync(after);chunk_seconds=time.monotonic()-t
        halfway,_=queued(base,32,kind,.5,sample,optimizer,cache,16)
        path=OUT/'performance_checkpoints'/f'{kind}_roundtrip.eqx'
        save_checkpoint(path,halfway);restored=load_checkpoint(path,seed)
        resumed,_=queued(restored,32,kind,.5,sample,optimizer,cache,16)
        values=dict(model_exact=exact(before['model'],after['model']),adam_exact=exact(before['opt_state'],after['opt_state']),
            prng_exact=exact(before['seeds'],after['seeds']),status_exact=bool(np.array_equal(strict_status,chunk_status)),
            model_max_abs=maxdiff(before['model'],after['model']),adam_max_abs=maxdiff(before['opt_state'],after['opt_state']),
            resume_exact=exact(after,resumed),steps=64)
        assert all(values[k] for k in ['model_exact','adam_exact','prng_exact','status_exact','resume_exact'])
        timings[kind]=dict(strict_seconds=strict_seconds,chunk_seconds=chunk_seconds,speedup=strict_seconds/chunk_seconds)
        metrics[kind]=values;selected[kind]=bool(strict_seconds/chunk_seconds>1.03)
        reason[kind]='exact equivalence and measured speedup >1.03' if selected[kind] else 'exact equivalence; chunk speedup <=1.03, use single-step acceptance'
        print('PERFORMANCE_KIND',kind,json.dumps(timings[kind]),'chunk_selected',selected[kind],flush=True)
    result=dict(passed=True,code_manifest_sha256=code_identity(),metrics=metrics,timings=timings,
        selected=selected,reason=reason,chunk_size=16,scientific_batch_size=32,
        compute_seconds=time.monotonic()-started,committed_training_steps=0)
    write_json(receipt,result);return result
