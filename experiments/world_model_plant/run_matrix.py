"""Run the four frozen prototype arms sequentially on Linux."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from .train import ROOT, verify_release, write_json


def run(dataset, properties, output, config, release, device):
    if platform.system() != 'Linux':
        raise RuntimeError('Run scientific fitting on Linux')
    verify_release(ROOT, release, config)
    task = json.loads(config.read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=False)
    state = dict(task_id=task['task_id'], state='running', runs=[],
                 started_utc=datetime.now(timezone.utc).isoformat())
    status_path = output / 'public_matrix_status.json'
    for seed in task['seeds']:
        for variant in task['variants']:
            name = f'{variant}_seed{seed}'
            entry = dict(variant=variant, seed=seed, state='running')
            state['runs'].append(entry)
            command = [sys.executable, '-m', 'experiments.world_model_plant.train',
                       '--dataset', str(dataset), '--properties', str(properties),
                       '--output', str(output / name), '--config', str(config),
                       '--release', str(release), '--variant', variant, '--seed', str(seed),
                       '--device', device]
            started = time.monotonic()
            with (output / f'{name}.private.log').open('w', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                entry['pid'] = process.pid
                while process.poll() is None:
                    entry['elapsed_seconds'] = round(time.monotonic() - started, 1)
                    entry['process_alive'] = True
                    state['checked_utc'] = datetime.now(timezone.utc).isoformat()
                    write_json(status_path, state)
                    if entry['elapsed_seconds'] >= task['timeout_seconds_per_arm']:
                        print(f'Hard timeout reached for {name}; stopping this arm.', flush=True)
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        entry['state'] = 'timed_out'
                        break
                    time.sleep(30)
                entry['exit_code'] = process.returncode
                entry['process_alive'] = False
                entry['elapsed_seconds'] = round(time.monotonic() - started, 1)
                if entry['state'] != 'timed_out':
                    entry['state'] = 'completed' if process.returncode == 0 else 'failed'
                write_json(status_path, state)
            if entry['state'] != 'completed':
                state['state'] = 'stopped_after_failure'
                state['finished_utc'] = datetime.now(timezone.utc).isoformat()
                write_json(status_path, state)
                return 1
    state['state'] = 'completed'
    state['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(status_path, state)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'properties', 'output', 'config', 'release'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    raise SystemExit(run(**vars(args)))
