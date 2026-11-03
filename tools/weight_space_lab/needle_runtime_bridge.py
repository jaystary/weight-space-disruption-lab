"""Replay one preserved numerical program with declared kernel bindings; no model call."""
import hashlib
import json
from pathlib import Path
import shutil
import time

from .contracts import file_sha, write_once
from .edit_loop import audit
from .sandbox import build, execute, JAIL, ROOT


def run():
    if not str(ROOT).startswith('/workspace/'):
        raise RuntimeError('remote workspace required')
    regpath = ROOT/'registrations/needle-runtime-bridge-01.json'
    reg = json.loads(regpath.read_text())
    previous = Path('/workspace/weight-space-needle-diagnostic-01b')
    original = json.loads((previous/'runs/needle-weight-diagnostic-01b/editor/answer.json').read_text())['code']
    if hashlib.sha256(original.encode()).hexdigest() != reg['program_sha256']:
        raise ValueError('retained numerical program changed')
    code = reg['prefix']+original
    audit(code)
    out = ROOT/'runs'/reg['id']
    out.mkdir(parents=True, exist_ok=False)
    write_once(out/'intent.json', {'registration_sha256': file_sha(regpath),
        'original_sha256': reg['program_sha256'], 'executed_sha256': hashlib.sha256(code.encode()).hexdigest(),
        'prefix_sha256': hashlib.sha256(reg['prefix'].encode()).hexdigest(), 'epoch': time.time(), 'limits': reg['limits']})
    source = previous/'snapshots'
    manifest = json.loads((source/'manifest.json').read_text())
    snapshots = ROOT/'snapshots'
    snapshots.mkdir(exist_ok=False)
    for spec in manifest['tensors']:
        path = source/spec['file']
        if file_sha(path) != spec['file_sha256']:
            raise ValueError('prior snapshot changed')
        shutil.copyfile(path, snapshots/spec['file'])
    write_once(snapshots/'manifest.json', manifest)
    build()
    shutil.copyfile(ROOT/'tools/weight_space_lab/edit_api.py', JAIL/'work/edit_api.py')
    (JAIL/'work/edit_api.py').chmod(0o444)
    rc = execute(code, out/'execution')
    result = {'id': reg['id'], 'program_returncode': rc, 'editor_calls': 0, 'gpu_calls': 0,
              'original_sha256': reg['program_sha256'], 'api_binding_correction': True,
              'stdout_bytes': (out/'execution/stdout.log').stat().st_size,
              'untested_patch_emitted': (out/'execution/patch.npz').exists(),
              'interpretation': 'Astra-authored numerical analysis; controller supplied missing API imports. No efficacy evidence.'}
    write_once(out/'aggregate-report.json', result)
    print(json.dumps(result), flush=True)
    if rc:
        raise SystemExit(1)


if __name__ == '__main__':
    run()
