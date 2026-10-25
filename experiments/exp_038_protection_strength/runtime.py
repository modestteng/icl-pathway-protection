"""Fixed C/P/D strength grid; unchanged scientific kernels, resumable dispatch."""
from common import *
import science as R
from performance import engine
import argparse, signal, traceback, shutil
from datetime import datetime, timezone

STOP = False


def stop(signum, frame):
    global STOP
    STOP = True


def cost(start, previous):
    return previous + time.monotonic() - start


def record_cost(start, previous, status):
    write_json(OUT/'compute_seconds.json', dict(compute_seconds=cost(start,previous),
        status=status, code_manifest_sha256=code_identity(),
        includes='scientific preparation, validation, training, evaluation and readback',
        excludes='agent work, transfer, installation and queue'))


def budget(start, previous):
    return STOP or cost(start,previous) >= SETTINGS['max_compute_seconds']


def reference_training_keys(seed):
    parent = Path(SETTINGS['reference_work'])
    verification = json.loads((WORK/'references/exp030/verification.json').read_text())
    path = parent/'outputs/s5/checkpoints'/f'A_{SETTINGS["end_stage"]:011d}.eqx'
    expected = verification['verified_file_sha256'].get(str(path.resolve()))
    assert expected is not None and sha(path)==expected, 'unregistered historical A checkpoint'
    ck = load_checkpoint(path,seed)
    leaves=[]
    with path.open('rb') as file:
        while file.tell()<path.stat().st_size:
            leaves.append(np.load(file,allow_pickle=False))
    names=['eval_model_seed','train_data_seed','train_model_seed']
    # The leaf stream is read independently again by the NumPy-only reader.
    candidates=[i for i,a in enumerate(leaves) if a.shape==(2,) and a.dtype==np.uint32]
    assert len(candidates)>=3
    indices=dict(zip(names,candidates[-3:]))
    for name,index in indices.items():
        assert np.array_equal(leaves[index],np.asarray(ck['seeds'][name])), (name,index)
    data={name:np.asarray(ck['seeds'][name]).tolist() for name in names}
    digest=hashlib.sha256(b''.join(np.asarray(ck['seeds'][name],np.uint32).tobytes()
        for name in ['train_data_seed','train_model_seed'])).hexdigest()
    write_json(OUT/'s5/reference_training_prng.json',dict(source_path=str(path),
        source_sha256=expected,seed_leaf_indices=indices,training_prng_sha256=digest,**data))


def train_arm(seed, arm, mechanism, performance, start, previous, shared):
    sd=OUT/seed['id'];complete=sd/'training_complete'/f'{arm}.json'
    kind=arm[0];rho=SETTINGS['relaxation'][arm]
    assert kind in ('C','P','D') and rho in (.1,.5,.75)
    if complete.exists():
        done=json.loads(complete.read_text())
        assert done['code_manifest_sha256']==code_identity() and done['rho']==rho
        assert done['end_stage']==SETTINGS['end_stage']
        assert sha(WORK/done['final_checkpoint_path'])==done['final_checkpoint_sha256']
        return
    assert mechanism['passed'] and mechanism['candidate_head']==0
    pointer=sd/'checkpoints'/f'{arm}_resume.json'
    stats=dict(steps=0,update_norm_ratio=0.,correction_norm_ratio=0.,conflicts=0.,near_zero_steps=0)
    last=-1
    if pointer.exists():
        saved=json.loads(pointer.read_text())
        assert saved['code_manifest_sha256']==code_identity() and saved['rho']==rho
        assert sha(WORK/saved['checkpoint_path'])==saved['checkpoint_sha256']
        ck=load_checkpoint(WORK/saved['checkpoint_path'],seed)
        stats,last=saved['stats'],saved['last_evaluated_iter']
    else:
        ck=base_checkpoint(seed)
    sample,optimizer,cache,mon,mids,confirm,cids,original=shared
    arm_start=time.monotonic()

    def save_resume():
        path=sd/'checkpoints'/f'{arm}_resume_{int(ck["iter"]):011d}.eqx'
        digest=save_checkpoint(path,ck)
        write_json(pointer,dict(iter=int(ck['iter']),stats=stats,last_evaluated_iter=last,
            checkpoint_path=str(path.relative_to(WORK)),checkpoint_sha256=digest,
            code_manifest_sha256=code_identity(),arm=arm,kind=kind,rho=rho))

    while True:
        i=int(ck['iter'])
        if i%SETTINGS['eval_every']==0 and last!=i:
            ck['seeds']['eval_model_seed'],_=jax.random.split(ck['seeds']['eval_model_seed'])
            save_npz(sd/'original'/f'{arm}_{i:011d}.npz',R.evaluate(ck['model'],original))
            if i%SETTINGS['monitor_every']==0 or i==SETTINGS['start_stage']:
                save_npz(sd/'monitor'/f'{arm}_{i:011d}.npz',R.evaluate(ck['model'],mon,mids))
            if i in SETTINGS['confirm_stages']:
                save_npz(sd/'confirm'/f'{arm}_{i:011d}.npz',R.evaluate(ck['model'],confirm,cids))
            assert int(np.asarray(ck['opt_state'][0].count))==i//BS
            assert all(np.isfinite(np.asarray(v)).all() for v in arrays(ck))
            path=sd/'checkpoints'/f'{arm}_{i:011d}.eqx'
            digest=save_checkpoint(path,ck)
            keys={name:np.asarray(value).tolist() for name,value in ck['seeds'].items()}
            write_json(sd/'evaluated'/f'{arm}_{i:011d}.json',dict(iter=i,
                checkpoint_sha256=digest,checkpoint_path=str(path.relative_to(WORK)),
                code_manifest_sha256=code_identity(),arm=arm,kind=kind,rho=rho,
                adam_count=i//BS,**keys))
            last=i
            save_resume()
            record_cost(start,previous,'training')
            write_json(sd/'progress.json',dict(seed=seed,arm=arm,kind=kind,rho=rho,
                iter=i,end=SETTINGS['end_stage'],compute_seconds=cost(start,previous),
                arm_current_process_seconds=time.monotonic()-arm_start,
                code_manifest_sha256=code_identity()))
            print('PROGRESS',arm,i,'rho',rho,'elapsed',round(cost(start,previous),2),flush=True)
        if budget(start,previous):
            save_resume();record_cost(start,previous,'safe_stop')
            write_json(OUT/'budget_stop.json',dict(arm=arm,iter=i,requested_stop=STOP,
                compute_seconds=cost(start,previous),max_compute_seconds=SETTINGS['max_compute_seconds']))
            raise SystemExit(3)
        if i==SETTINGS['end_stage']:
            break
        offset=(i-SETTINGS['start_stage'])//BS
        count=min(engine.CHUNK if performance['selected'][kind] else 1,
            (SETTINGS['end_stage']-i)//BS,(SETTINGS['eval_every']-i%SETTINGS['eval_every'])//BS)
        if offset in SETTINGS['diagnostic_update_offsets']:
            count=1
        else:
            future=[v-offset for v in SETTINGS['diagnostic_update_offsets'] if v>offset]
            if future:count=min(count,min(future))
        assert 1<=count<=engine.CHUNK
        try:
            candidate,status,audit=engine.queue_steps(ck,count,kind,rho,0,sample,optimizer,cache)
            if audit is not None:
                stage,old,new,raw,gradients,px,py=audit
                R.audit_update(seed,arm,stage,old,new,raw,gradients,px,py,mon,0,kind)
        except Exception as error:
            save_resume()
            write_json(sd/'failure'/f'{arm}.json',dict(iter=i,error=str(error),
                uncommitted_chunk_rolled_back=True,code_manifest_sha256=code_identity()))
            raise
        for row in status:
            stats['update_norm_ratio']+=float(row[3]);stats['correction_norm_ratio']+=float(row[4])
            stats['conflicts']+=float(row[5]);stats['near_zero_steps']+=int(float(row[3])<.01)
        stats['steps']+=count
        ck=candidate
    assert stats['steps']==175000
    key_record={name:np.asarray(ck['seeds'][name]).tolist()
        for name in ['train_data_seed','train_model_seed','eval_model_seed']}
    expected=json.loads((sd/'reference_training_prng.json').read_text())
    for name in ['train_data_seed','train_model_seed']:
        assert key_record[name]==expected[name], 'training PRNG diverged from historical A'
    key_digest=hashlib.sha256(b''.join(np.asarray(ck['seeds'][name],np.uint32).tobytes()
        for name in ['train_data_seed','train_model_seed'])).hexdigest()
    path=sd/'checkpoints'/f'{arm}_{i:011d}.eqx'
    p=provenance()
    write_json(complete,dict(passed=True,seed=seed,arm=arm,kind=kind,rho=rho,
        constraint_count={'C':2,'P':1,'D':3}[kind],end_stage=i,steps=stats['steps'],
        final_checkpoint_path=str(path.relative_to(WORK)),final_checkpoint_sha256=sha(path),
        update_statistics_sum=stats,update_statistics=dict(
            mean_update_norm_ratio=stats['update_norm_ratio']/175000,
            mean_correction_norm_ratio=stats['correction_norm_ratio']/175000,
            conflict_rate=stats['conflicts']/175000,near_zero_rate=stats['near_zero_steps']/175000),
        inputs=p['inputs'],source=p['source'],new_inputs=p['new_inputs'],
        training_prng=key_record,training_prng_sha256=key_digest,
        compute_seconds=cost(start,previous),arm_current_process_seconds=time.monotonic()-arm_start,
        code_manifest_sha256=code_identity(),execution_engine='queued_original_kernels' if performance['selected'][kind] else 'single_step_original_kernels'))
    print('ARM_COMPLETE',arm,flush=True)


def run():
    from datasets import prepare
    from prepare_run import validate_references
    from preflight import preflight, validate_performance
    prevfile=OUT/'compute_seconds.json'
    previous=json.loads(prevfile.read_text())['compute_seconds'] if prevfile.exists() else 0.
    start=time.monotonic()
    try:
        verify_inputs();validate_references();prepare()
        p=provenance();p['new_inputs']={str(f.relative_to(WORK)):sha(f) for f in sorted((OUT/'data').glob('*.h5'))}
        write_json(OUT/'provenance.json',p)
        print('INPUTS_VERIFIED',code_identity(),flush=True)
        preflight()
        seed=SETTINGS['training_seeds'][0]
        reference_training_keys(seed)
        mechfile=OUT/'s5/mechanism_confirmation.json'
        mechanism=json.loads(mechfile.read_text()) if mechfile.exists() else R.validate_mechanism(seed)
        assert mechanism['passed'] and mechanism['candidate_head']==0
        assert mechanism['code_manifest_sha256']==code_identity()
        performance=validate_performance()
        protect,_=load_dataset('protection');mon,mids=load_dataset('monitor');confirm,cids=load_dataset('confirm')
        original=R.original_data()
        mon,confirm,original=[engine.stage_on_gpu(v) for v in (mon,confirm,original)]
        _,optimizer=template(seed)
        shared=(make_sampler(seed),optimizer,engine.cache_protection(protect),mon,mids,confirm,cids,original)
        for arm in SETTINGS['arms']:
            train_arm(seed,arm,mechanism,performance,start,previous,shared)
        record_cost(start,previous,'training_complete')
        import subprocess
        for name in ['analyze.py','verify_results.py']:
            remaining=SETTINGS['max_compute_seconds']-cost(start,previous)
            if remaining<=0:raise SystemExit(3)
            subprocess.run([sys.executable,str(WORK/name)],check=True,timeout=remaining)
        check=json.loads((OUT/'verification.json').read_text())
        assert check['passed'] and not check['failures']
        write_json(OUT/'runtime_complete.json',dict(passed=True,arms=SETTINGS['arms'],
            code_manifest_sha256=code_identity(),compute_seconds=cost(start,previous),
            verification_sha256=sha(OUT/'verification.json'),analysis_sha256=sha(OUT/'analysis.json'),
            scope='one training seed; matched fixed-grid continuation plus independent saved-output readback'))
    except Exception as error:
        attempt=WORK/'attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        attempt.mkdir(parents=True)
        write_json(attempt/'failure.json',dict(error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),code_manifest_sha256=code_identity(),
            compute_seconds=cost(start,previous),automatic_retry=False,scientific_conditions_changed=False))
        for name in json.loads((WORK/'code_manifest.json').read_text()):
            dest=attempt/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(WORK/name,dest)
        shutil.copyfile(WORK/'code_manifest.json',attempt/'source/code_manifest.json')
        raise
    finally:
        record_cost(start,previous,'runner_exit')


if __name__=='__main__':
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run']);parser.parse_args()
    run()
