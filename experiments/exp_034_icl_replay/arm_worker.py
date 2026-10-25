"""Keep each CUDA context warm for gates, one bounded timing, and training."""
import argparse
import os
from pathlib import Path
import signal
import time
import traceback

from common import OUT, WORK, SETTINGS, code_identity, write_json
import runtime
from gpu_preflight import preflight, benchmark


def await_file(path):
    while not path.exists():
        if runtime.STOP or (path.parent / 'ABORT').exists():
            raise SystemExit(3)
        if time.time() >= float(os.environ.get('ICL034_DEADLINE_EPOCH', 'inf')):
            raise SystemExit(3)
        time.sleep(.1)


def work(arm, session):
    preflight(arm)
    write_json(session / (arm + '_ready.json'), {'passed': True, 'arm': arm,
               'code_manifest_sha256': code_identity()})
    # One bounded comparison: serial time of both arms, then their concurrent
    # time, all within these already-warm contexts. No training state is saved.
    for label in ('serial', 'parallel'):
        await_file(session / (label + '_' + arm + '_go'))
        os.environ['ICL034_BENCHMARK_LABEL'] = label
        value = benchmark(arm, 128)
        write_json(session / (label + '_' + arm + '_done.json'), value)
    await_file(session / ('train_' + arm + '_go'))
    runtime.run(arm, ['s5'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=['REPLAY', 'L2'])
    parser.add_argument('--session', required=True, type=Path)
    args = parser.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, runtime.request_stop)
    try:
        work(args.arm, args.session)
    except BaseException as error:
        write_json(args.session / (args.arm + '_worker_exit.json'), {
            'arm': args.arm, 'error_type': type(error).__name__, 'error': str(error),
            'traceback': traceback.format_exc(), 'automatic_retry': False})
        raise
