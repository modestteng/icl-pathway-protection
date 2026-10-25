"""Run an unchanged scientific worker with an explicit supplemental seed registry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    original = args.original.resolve()
    config = json.loads((HERE / 'supplement.json').read_text())
    os.environ['ICL042_OUTPUTS'] = str(HERE / 'outputs')
    os.environ['ICL042_ASSETS'] = str(original / 'assets')
    sys.path.insert(0, str(original))
    import common as c
    import worker as w
    assert c.code_identity() == config['original_manifest_sha256']
    assert c.SETTINGS['long_seeds'] == ['s18', 's19', 's20']
    c.SETTINGS['long_seeds'] = config['effective_settings_override']['long_seeds'].copy()
    parts = args.job.split('_')
    sid, arm = parts[0], parts[1]
    assert sid in config['seeds'] and arm in config['arms']
    assert args.job in [f'{sid}_{arm}_40M', f'{sid}_{arm}_40M_diagnostic']
    directory = c.OUT / 'jobs' / args.job
    directory.mkdir(parents=True, exist_ok=True)
    identity = c.sha(HERE / 'manifest.json')
    for name, expected in json.loads((HERE / 'manifest.json').read_text()).items():
        assert c.sha(HERE / name) == expected, name
    signal.signal(signal.SIGTERM, w.request_stop)
    signal.signal(signal.SIGINT, w.request_stop)
    started = time.monotonic()
    try:
        environment = c.verify_inputs()
        environment.update(supplement_manifest_sha256=identity,
                           effective_settings_override=config['effective_settings_override'])
        c.write_json(directory / 'input_environment.json', environment)
        seed = c.seed_spec(sid)
        result = (w.diagnostic(seed, arm, config['end']) if args.job.endswith('_diagnostic')
                  else w.train(seed, arm, config['end'], directory))
        c.write_json(directory / 'complete.json', dict(passed=True, result=result,
            job=args.job, code_manifest_sha256=c.code_identity(),
            supplement_manifest_sha256=identity, elapsed_seconds=time.monotonic()-started,
            utc=w.now()))
    except SystemExit as error:
        c.write_json(directory / 'stopped.json', dict(exit_code=error.code, utc=w.now(),
            checkpoint_preserved=True, automatic_retry=False))
        raise
    except Exception:
        c.write_json(directory / 'failure.json', dict(traceback=traceback.format_exc(),
            utc=w.now(), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
