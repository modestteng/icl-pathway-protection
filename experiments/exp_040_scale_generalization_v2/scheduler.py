"""Dependency scheduler for a Linux CUDA host. Launches isolated worker processes.

No JAX import. Each job is ``python worker.py --job ...`` with its own CUDA
context, ``CUDA_VISIBLE_DEVICES`` and memory fraction. A job is done when
``outputs/jobs/<name>/complete.json`` exists for the sealed code. Failed jobs are
never retried automatically (rerun with ``--retry-failed`` after inspection);
dependants of a failed job are not launched. ``outputs/STOP`` stops launching
and asks running workers to save at their next 100k boundary.

Usage:
  python scheduler.py --plan                      # print the job graph
  python scheduler.py --until validate            # data + validations only
  python scheduler.py --workers-per-gpu 3         # full run
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

WORK = Path(__file__).resolve().parent
SETTINGS = json.loads((WORK / 'settings.json').read_text())
OUTPUTS = Path(os.environ.get('ICL040_OUTPUTS', str(WORK / 'outputs'))).resolve()
ORDER = ['data', 'validate', 'normal', 'mechanism', 'arm']
ARM_ORDER = ['P25', 'PR25', 'C25', 'P75', 'C75']


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def code_identity():
    return sha(WORK / 'code_manifest.json')


def job_name(job, arch=None, seed=None, arm=None):
    return '_'.join(v for v in [job, arch, seed, arm] if v)


def job_state(name):
    d = OUTPUTS / 'jobs' / name
    if (d / 'complete.json').exists():
        record = json.loads((d / 'complete.json').read_text())
        return 'done' if record.get('code_manifest_sha256') == code_identity() else 'stale'
    if (d / 'failure.json').exists():
        return 'failed'
    if (d / 'stopped.json').exists():
        return 'stopped'
    return 'pending'


def timing(arch, seed):
    p = OUTPUTS / arch / seed / 'timing.json'
    return json.loads(p.read_text()) if p.exists() else None


def graph(arches, seeds):
    """Yield (name, job, arch, seed, arm, deps) for the static plan."""
    jobs = [('data', 'data', None, None, None, [])]
    for arch in arches:
        jobs.append((job_name('validate', arch), 'validate', arch, None, None, ['data']))
    for seed in seeds:
        for arch in arches:
            jobs.append((job_name('normal', arch, seed), 'normal', arch, seed, None, [job_name('validate', arch)]))
    for seed in seeds:
        for arch in arches:
            jobs.append((job_name('mechanism', arch, seed), 'mechanism', arch, seed, None,
                         [job_name('normal', arch, seed)]))
    for arm in ARM_ORDER:
        for seed in seeds:
            for arch in reversed(arches):
                jobs.append((job_name('arm', arch, seed, arm), 'arm', arch, seed, arm,
                             [job_name('mechanism', arch, seed)]))
    return jobs


def applicable(job, arch, seed):
    """mechanism/arm jobs exist only for seeds the frozen timing rule admits."""
    if job in ('mechanism', 'arm'):
        lock = timing(arch, seed)
        return None if lock is None else bool(lock['applicable'])
    return True


def gpus():
    if os.environ.get('ICL040_GPUS'):
        return [g for g in os.environ['ICL040_GPUS'].split(',') if g != '']
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                             capture_output=True, text=True, check=True).stdout.split()
        return out
    except Exception:
        return ['0']


def free_gib(path):
    return shutil.disk_usage(path).free / 2 ** 30


def launch(name, job, arch, seed, arm, gpu, fraction, python):
    log_dir = OUTPUTS / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [python, '-u', str(WORK / 'worker.py'), '--job', job]
    if seed:
        cmd += ['--seed', seed]
    if arm:
        cmd += ['--arm', arm]
    env = dict(os.environ, ICL040_ARCH=arch or 'base', CUDA_VISIBLE_DEVICES=str(gpu),
               XLA_PYTHON_CLIENT_MEM_FRACTION=f'{fraction:.3f}', ICL040_OUTPUTS=str(OUTPUTS))
    stream = (log_dir / (name + '.log')).open('a')
    stream.write(f'\n===== launch {now()} gpu={gpu} fraction={fraction:.3f}\n')
    stream.flush()
    proc = subprocess.Popen(cmd, cwd=str(WORK), env=env, stdout=stream, stderr=subprocess.STDOUT)
    return dict(proc=proc, stream=stream, gpu=gpu, started=time.time(), name=name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers-per-gpu', type=int, default=SETTINGS['execution']['default_workers_per_gpu'])
    parser.add_argument('--max-workers', type=int, default=None)
    parser.add_argument('--until', choices=ORDER, default='arm')
    parser.add_argument('--arch', action='append', choices=list(SETTINGS['architectures']))
    parser.add_argument('--seed', action='append')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--poll', type=float, default=20.0)
    args = parser.parse_args()
    arches = args.arch or SETTINGS['architecture_order']
    seeds = args.seed or [s['id'] for s in SETTINGS['training_seeds']]
    plan = [j for j in graph(arches, seeds) if ORDER.index(j[1]) <= ORDER.index(args.until)]
    if args.plan:
        for name, job, arch, seed, arm, deps in plan:
            print(f'{name:40s} deps={deps} state={job_state(name)}')
        print(len(plan), 'jobs')
        return
    devices = gpus()
    slots = len(devices) * args.workers_per_gpu
    if args.max_workers:
        slots = min(slots, args.max_workers)
    fraction = SETTINGS['execution']['gpu_memory_total_fraction'] / args.workers_per_gpu
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    running = {}
    if args.retry_failed:
        for name, *_ in plan:
            failure = OUTPUTS / 'jobs' / name / 'failure.json'
            if failure.exists():
                failure.rename(failure.with_name(f'failure_{int(time.time())}.json'))
    print('SCHEDULER_START', now(), 'gpus', devices, 'slots', slots, 'fraction', fraction, flush=True)
    while True:
        stop = (OUTPUTS / 'STOP').exists()
        for name, item in list(running.items()):
            code = item['proc'].poll()
            if code is not None:
                item['stream'].close()
                print('JOB_EXIT', now(), name, code, job_state(name),
                      'seconds=%.1f' % (time.time() - item['started']), flush=True)
                del running[name]
        states = {name: job_state(name) for name, *_ in plan}
        blocked, waiting, ready = [], [], []
        for name, job, arch, seed, arm, deps in plan:
            if states[name] == 'done' or name in running:
                continue
            if states[name] == 'failed':
                blocked.append(name)
                continue
            dep_states = [states.get(d) for d in deps]
            if any(s == 'failed' for s in dep_states):
                blocked.append(name)
                continue
            if not all(s == 'done' for s in dep_states):
                waiting.append(name)
                continue
            ok = applicable(job, arch, seed)
            if ok is False:
                states[name] = 'not_applicable'
                continue
            if ok is None:
                waiting.append(name)
                continue
            ready.append((name, job, arch, seed, arm))
        status = dict(utc=now(), running={n: dict(gpu=i['gpu'], seconds=time.time() - i['started'])
                                          for n, i in running.items()},
                      counts={s: sum(1 for v in states.values() if v == s) for s in set(states.values())},
                      ready=len(ready), waiting=len(waiting), blocked=blocked, stop=stop,
                      free_gib=free_gib(OUTPUTS), slots=slots)
        (OUTPUTS / 'scheduler_status.json').write_text(json.dumps(status, indent=2) + '\n')
        low_disk = free_gib(OUTPUTS) < SETTINGS['execution']['min_free_disk_gib']
        if low_disk and ready:
            print('LOW_DISK_NO_NEW_JOBS', '%.1f GiB free' % free_gib(OUTPUTS), flush=True)
        if not running and (not ready or stop or low_disk):
            print('SCHEDULER_IDLE', json.dumps(status['counts']), 'blocked', blocked, 'stop', stop,
                  'low_disk', low_disk, flush=True)
            break
        while ready and len(running) < slots and not stop and not low_disk:
            name, job, arch, seed, arm = ready.pop(0)
            load = {g: sum(1 for i in running.values() if i['gpu'] == g) for g in devices}
            gpu = min(devices, key=lambda g: (load[g], devices.index(g)))
            running[name] = launch(name, job, arch, seed, arm, gpu, fraction, args.python)
            print('JOB_LAUNCH', now(), name, 'gpu', gpu, flush=True)
            if job in ('data',):
                break
        time.sleep(args.poll)
    print('SCHEDULER_END', now(), flush=True)


if __name__ == '__main__':
    main()
