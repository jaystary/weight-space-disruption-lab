"""One registered CPU weight-inspection round; no target-model forward pass."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .contracts import file_sha, write_once
from .edit_loop import audit, SCHEMA
from .model_client import call
from .research_prompt import render
from .sandbox import build, execute, JAIL

ROOT = Path(__file__).resolve().parents[2]


def snapshot_names(registration):
    return [f'model.language_model.layers.{layer}.{module}'
            for layer in registration['layers'] for module in registration['modules']]


def run():
    if not str(ROOT).startswith('/workspace/'):
        raise RuntimeError('remote workspace required')
    import numpy as np
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    registration_path = ROOT/'registrations/needle-weight-diagnostic-01b.json'
    registration = json.loads(registration_path.read_text())
    out = ROOT/'runs'/registration['id']
    out.mkdir(parents=True, exist_ok=False)
    out.chmod(0o700)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'tools', 'tests', 'registrations', 'prompts'],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    start = time.monotonic()
    write_once(out/'intent.json', {'source_commit': source, 'registration_sha256': file_sha(registration_path),
                                 'started_epoch': time.time(), 'limits': registration['limits']})
    report = {'id': registration['id'], 'source_commit': source, 'status': 'failed',
              'gpu_calls': 0, 'scored_cases': 0, 'editor_calls': 0}
    try:
        model_path = Path(snapshot_download(registration['model'], revision=registration['revision'],
                         cache_dir='/workspace/weight-space-lab/model-cache', local_files_only=True,
                         allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt']))
        snapshots = ROOT/'snapshots'
        snapshots.mkdir(exist_ok=False)
        names = snapshot_names(registration)
        specs = []
        for shard in sorted(model_path.glob('*.safetensors')):
            with safe_open(shard, framework='pt', device='cpu') as archive:
                for name in names:
                    if name not in archive.keys():
                        continue
                    import torch
                    tensor = archive.get_tensor(name)
                    if tensor.dtype != torch.bfloat16:
                        raise ValueError('expected immutable BF16 tensor')
                    bits = tensor.contiguous().view(torch.uint16).numpy()
                    path = snapshots/f'tensor-{len(specs)}.npy'
                    np.save(path, bits, allow_pickle=False)
                    specs.append({'name': name, 'shape': list(bits.shape), 'dtype': 'bfloat16',
                                  'sha256': hashlib.sha256(bits.tobytes()).hexdigest(),
                                  'file': path.name, 'file_sha256': file_sha(path)})
        if sorted(s['name'] for s in specs) != sorted(names):
            raise ValueError('missing or duplicate permitted tensor')
        manifest = {'schema': 'WSPTensors/1', 'checkpoint': registration['model'],
                    'revision': registration['revision'], 'tensors': specs}
        write_once(snapshots/'manifest.json', manifest)
        build()
        api = (ROOT/'tools/weight_space_lab/edit_api.py').read_text()
        (JAIL/'work/edit_api.py').write_text(api)
        (JAIL/'work/edit_api.py').chmod(0o444)
        context = {'stage': 'numerical inspection only; GPU occupied by frozen prior evaluation',
                   'manifest': manifest, 'available': ['load(name)', 'SPECS', 'numpy', 'print'],
                   'unavailable': ['forward passes', 'activations', 'attention traces', 'examples', 'final data'],
                   'task': 'Inspect actual tensors using executable Python. Print compact numerical evidence and propose which activation traces would distinguish retrieval failure from memory-write failure. Do not assert a semantic mechanism from norms alone. No save() needed yet.',
                   'response': 'Return exactly hypothesis and code strings. Code runs once in the isolated CPU jail; returned output informs later rounds.'}
        prompt = render(api, context, registration)
        report['editor_calls'] = 1
        answer, receipt = call(prompt, SCHEMA, out/'editor', model=registration['editor_model'],
                               effort=registration['editor_effort'], timeout=480,
                               executable='/workspace/tools/codex')
        write_once(out/'code-audit.json', audit(answer['code']))
        rc = execute(answer['code'], out/'execution')
        report.update(status='completed' if rc == 0 else 'program_failed', program_returncode=rc,
                      requested_model=receipt['model'], reported_models=receipt['reported_models'],
                      usage_known=receipt['usage_known'], usage=receipt['usage'],
                      snapshot_tensors=len(specs), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                      program_sha256=hashlib.sha256(answer['code'].encode()).hexdigest(),
                      untested_patch_emitted=(out/'execution/patch.npz').exists(),
                      numerical_output_bytes=(out/'execution/stdout.log').stat().st_size)
    except Exception as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error)[:400]}
        receipt_path = out/'editor/receipt.json'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            report.update(usage_known=receipt['usage_known'], usage=receipt['usage'],
                          requested_model=receipt['model'], reported_models=receipt['reported_models'])
    report['seconds'] = time.monotonic()-start
    write_once(out/'aggregate-report.json', report)
    print(json.dumps(report), flush=True)
    if report['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    run()
