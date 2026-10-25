"""Detached serial runner and resource protection. Standard library only."""
import json
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from io_utils import ROOT, OUT, CFG, sha, write_json, verify_package

ACTIVE = None


def processes_in_group(pgid):
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])==pgid and fields[0]!='Z':
                rss=0
                for line in (p/'status').read_text().splitlines():
                    if line.startswith('VmRSS:'):
                        rss=int(line.split()[1])
                found.append((int(p.name),rss))
        except (OSError,ValueError,IndexError):
            pass
    return found


def memory_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])
    raise RuntimeError('cannot inspect available memory')


def main():
    global ACTIVE
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume-from',choices=['prepare','encode','evaluate_A','evaluate_P25','verify'],default='prepare')
    parser.add_argument('--stop-after', choices=['prepare'])
    args_cli=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    verify_package()
    if (OUT/'runtime_complete.json').exists():
        raise RuntimeError('completed experiment exists; no automatic rerun')
    began=time.monotonic()
    phases=[('prepare',['worker.py','prepare']),('encode',['worker.py','encode']),
            ('evaluate_A',['worker.py','evaluate','--arm','A']),
            ('evaluate_P25',['worker.py','evaluate','--arm','P25']),
            ('verify',['analyze_verify.py'])]
    first=[p[0] for p in phases].index(args_cli.resume_from)
    receipts=[]
    for name,_ in phases[:first]:
        receipt=json.loads((logs/(name+'.exit.json')).read_text())
        assert receipt['exit_code']==0 and not receipt['remaining_group_members']
        receipts.append(receipt)
    if first:
        data=json.loads((OUT/'data_manifest.json').read_text())
        assert sha(OUT/'selected_images.h5')==data['selected_images_sha256']
        assert sha(OUT/'episodes.npz')==data['episodes_sha256']
    env=dict(os.environ,JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',
             OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MPLBACKEND='Agg')
    with (logs/'resources.jsonl').open('a',buffering=1) as resources:
        for name,args in phases[first:]:
            if (logs/(name+'.exit.json')).exists():
                raise RuntimeError('prior phase receipt exists; inspect instead of automatic retry: '+name)
            start=time.monotonic()
            with (logs/(name+'.log')).open('w',buffering=1) as log:
                p=subprocess.Popen([CFG['python'],'-u']+args,cwd=ROOT,env=env,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                ACTIVE = p
                write_json(OUT/'status.json',dict(phase=name,pid=p.pid,state='running',utc=datetime.now(timezone.utc).isoformat()))
                low=0;reason=None;maxrss=0;minmem=1<<60
                while p.poll() is None:
                    members=processes_in_group(p.pid)
                    rss=sum(v for _,v in members);available=memory_available()
                    maxrss=max(maxrss,rss);minmem=min(minmem,available)
                    low=low+1 if available<CFG['limits']['min_available_mib']*1024 else 0
                    try:
                        gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],
                                           capture_output=True,text=True,timeout=10).stdout.strip()
                    except subprocess.TimeoutExpired:
                        gpu='observation_timeout'
                    resources.write(json.dumps(dict(phase=name,pid=p.pid,seconds=time.monotonic()-start,
                        rss_tree_kib=rss,available_kib=available,gpu=gpu))+'\n')
                    if low>=2 or rss>CFG['limits']['max_child_rss_mib']*1024:
                        reason='resource_failure'
                    if time.monotonic()-start>CFG['limits']['phase_timeout_seconds'] or time.monotonic()-began>CFG['limits']['total_timeout_seconds']:
                        reason='timeout'
                    if reason:
                        os.killpg(p.pid,signal.SIGTERM)
                        try:p.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(p.pid,signal.SIGKILL);p.wait()
                        break
                    time.sleep(CFG['limits']['poll_seconds'])
                code=p.wait()
                for _ in range(10):
                    if not processes_in_group(p.pid):break
                    time.sleep(.2)
                remaining=processes_in_group(p.pid)
                ACTIVE = None
                receipt=dict(phase=name,exit_code=code,stop_reason=reason,seconds=time.monotonic()-start,
                             max_rss_tree_kib=maxrss,min_available_kib=minmem,remaining_group_members=remaining)
                write_json(logs/(name+'.exit.json'),receipt);receipts.append(receipt)
                if code or reason or remaining:
                    write_json(OUT/'status.json',dict(state='failed',**receipt))
                    raise RuntimeError('phase stopped: '+json.dumps(receipt))
            if args_cli.stop_after == name:
                write_json(OUT/'status.json',dict(state='prepared',phase=name))
                return
    verify_package()
    files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size)
           for p in OUT.iterdir() if p.is_file() and p.name not in ['status.json','runtime_complete.json','output_manifest.json']}
    write_json(OUT/'output_manifest.json',files)
    write_json(OUT/'runtime_complete.json',dict(status='completed',exit_code=0,phases=receipts,
        elapsed_seconds=time.monotonic()-began,utc=datetime.now(timezone.utc).isoformat(),
        code_manifest_sha256=sha(ROOT/'code_manifest.json')))
    write_json(OUT/'status.json',dict(state='completed',phase='all'))


if __name__=='__main__':
    try:
        main()
    except BaseException:
        if ACTIVE is not None and ACTIVE.poll() is None:
            os.killpg(ACTIVE.pid,signal.SIGTERM)
            try:ACTIVE.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(ACTIVE.pid,signal.SIGKILL);ACTIVE.wait()
        OUT.mkdir(exist_ok=True)
        (OUT/'controller_failure.txt').write_text(traceback.format_exc())
        raise
