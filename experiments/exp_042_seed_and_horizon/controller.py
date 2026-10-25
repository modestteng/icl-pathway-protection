"""Serial process supervisor. No JAX, model, or CUDA import in this parent."""
from plan import *
import argparse
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone


def utc():
    return datetime.now(timezone.utc).isoformat()


def completed(name):
    path = OUTPUTS / 'jobs' / name / 'complete.json'
    if not path.exists():
        return False
    record = read_json(path)
    if record['code_manifest_sha256'] != code_identity() or not record['passed']:
        raise RuntimeError('stale or invalid job receipt: ' + name)
    return True


def memory():
    info = Path('/proc/meminfo').read_text().splitlines()
    return next(int(v.split()[1]) / 1024 for v in info if v.startswith('MemAvailable:'))


def rss_mib(pid):
    try:
        lines = Path(f'/proc/{pid}/status').read_text().splitlines()
        return next(int(v.split()[1]) / 1024 for v in lines if v.startswith('VmRSS:'))
    except (FileNotFoundError, StopIteration):
        return 0.


def run_child(job, command=None):
    directory = OUTPUTS / 'jobs' / job['name']
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'failure.json').exists() or (directory / 'resource_failure.json').exists():
        raise RuntimeError('failed job requires explicit inspection; no automatic retry: ' + job['name'])
    if (directory / 'exit.json').exists():
        history = directory / 'attempt_history' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        history.mkdir(parents=True)
        for old in list(directory.iterdir()):
            if old.is_file():
                old.replace(history / old.name)
    env = dict(os.environ, PYTHONUNBUFFERED='1', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_PYTHON_CLIENT_MEM_FRACTION=str(SETTINGS['execution']['gpu_memory_fraction']))
    if job['kind'] in ['data', 'select', 'short_complete', 'analyze', 'verify']:
        env['JAX_PLATFORMS'] = 'cpu'
        env['CUDA_VISIBLE_DEVICES'] = ''
    else:
        env['JAX_PLATFORMS'] = 'cuda'
        env['CUDA_VISIBLE_DEVICES'] = os.environ.get('ICL042_GPU', '0')
    command = command or [sys.executable, '-u', str(WORK / 'worker.py'), '--job', job['name']]
    started = time.monotonic()
    low, peak, reason, last_gpu_check, gpu = 0, 0., None, 0., None
    with (directory / 'stdout.log').open('a') as output:
        proc = subprocess.Popen(command, cwd=WORK, env=env, stdout=output, stderr=subprocess.STDOUT,
                                start_new_session=True)
        write_json(directory / 'started.json', dict(pid=proc.pid, utc=utc(), job=job,
            code_manifest_sha256=code_identity(), command=command, controller_has_cuda_context=False))
        try:
            with (directory / 'resources.jsonl').open('a') as log:
                while proc.poll() is None:
                    available, rss = memory(), rss_mib(proc.pid)
                    peak = max(peak, rss)
                    if job['kind'] not in ['data', 'select', 'short_complete', 'analyze', 'verify'] and time.monotonic() - last_gpu_check >= 10:
                        try:
                            observation = subprocess.run(['nvidia-smi', '-i', os.environ.get('ICL042_GPU', '0'),
                                '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=2)
                            gpu = observation.stdout.strip() if observation.returncode == 0 else None
                        except (FileNotFoundError, subprocess.TimeoutExpired):
                            gpu = None
                        last_gpu_check = time.monotonic()
                    log.write(json.dumps(dict(utc=utc(), pid=proc.pid, available_mib=available, rss_mib=rss,
                                             gpu_utilization_and_memory=gpu)) + '\n')
                    log.flush()
                    limits = SETTINGS['execution']
                    low = low + 1 if available < limits['min_available_memory_mib'] else 0
                    if low >= limits['memory_low_consecutive'] or rss > limits['max_worker_rss_mib']:
                        reason = 'resource_guard'
                        # Preserve the previous complete checkpoint even if a sudden allocation
                        # prevents this worker from reaching its next 100k save boundary.
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                        break
                    time.sleep(limits['memory_poll_seconds'])
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait()
            raise
    result = dict(exit_code=proc.returncode, elapsed_seconds=time.monotonic() - started,
                  peak_rss_mib=peak, reason=reason, utc=utc(), real_exit_confirmed=True)
    write_json(directory / 'exit.json', result)
    if reason:
        write_json(directory / 'resource_failure.json', result)
        raise RuntimeError('resource failure; saved results retained; no retry')
    if proc.returncode != 0:
        if proc.returncode != 3:
            original = read_json(directory / 'failure.json') if (directory / 'failure.json').exists() else {}
            write_json(directory / 'failure.json', dict(result, worker_failure=original))
        raise SystemExit(proc.returncode)
    if job['kind'] in ['analyze', 'verify']:
        write_json(directory / 'complete.json', dict(passed=True, code_manifest_sha256=code_identity(), **result))
    if not completed(job['name']):
        raise RuntimeError('process exited without a complete receipt: ' + job['name'])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--until', choices=['validate', 'short', 'full'], default='full')
    args = parser.parse_args()
    jobs = graph()
    if args.plan:
        for job in jobs:
            print(job['name'], 'depends_on=' + ','.join(job['deps']))
        print(len(jobs), 'jobs; one child process at a time; no computations executed')
        return
    if sys.platform != 'linux':
        raise RuntimeError('execution is reserved for the user-provided Linux CUDA server')
    verify_code()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    # Exclusive lock prevents two controllers from advancing the same checkpoints.
    import fcntl
    lock = (OUTPUTS / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    for job in jobs:
        if args.until == 'validate' and job['kind'] not in ['data', 'validate']:
            continue
        if args.until == 'short' and job.get('budget') == 40000000:
            continue
        if (OUTPUTS / 'STOP').exists():
            raise SystemExit(3)
        if completed(job['name']):
            continue
        if not all(completed(dep) for dep in job['deps']):
            raise RuntimeError('job dependency not complete: ' + job['name'])
        if shutil.disk_usage(OUTPUTS).free / 2 ** 30 < SETTINGS['execution']['min_free_disk_gib']:
            raise RuntimeError('low disk; no new child started')
        print('START', utc(), job['name'], flush=True)
        result = run_child(job)
        print('EXIT', utc(), job['name'], result['elapsed_seconds'], flush=True)
    if args.until == 'full':
        for name in ['analyze', 'verify']:
            job = dict(name=name, kind=name)
            if not completed(name):
                filename = 'analyze.py' if name == 'analyze' else 'verify_results.py'
                run_child(job, [sys.executable, str(WORK / filename)])
        verification = read_json(OUTPUTS / 'verification.json')
        assert verification['passed']
        write_json(OUTPUTS / 'runtime_complete.json', dict(passed=True, code_manifest_sha256=code_identity(),
            verification_sha256=sha(OUTPUTS / 'verification.json'), analysis_sha256=sha(OUTPUTS / 'analysis.json'),
            latest_controller_seconds=time.monotonic() - started,
            recorded_job_seconds=sum(read_json(p)['elapsed_seconds'] for p in (OUTPUTS / 'jobs').glob('**/exit.json')),
            scope='job time recorded in saved exit receipts; installation/transfer excluded', utc=utc()))


if __name__ == '__main__':
    main()
