"""Remote CPU integration check on an unscored transformer. Run once in an isolated source copy."""
import hashlib
import json
import os
from pathlib import Path

from .contracts import file_sha, write_once
from .edit_loop import ALLOWED_IMPORTS, audit
from .needle_findings import collect_findings
from .needle_loop import compact_history, editor_interface
from .needle_worker import NeedleWorker
from .research_prompt import render
from .sandbox import JAIL, build, execute


def run():
    import numpy as np
    import torch

    root = Path(__file__).resolve().parents[2]
    if not root.is_relative_to('/workspace/') or os.geteuid() != 0 or torch.cuda.is_available():
        raise RuntimeError('run on the authorized worker with CUDA hidden in a new source copy')
    regpath = root/'registrations/editor-interface-smoke-01.json'
    registration = json.loads(regpath.read_text())
    out = root/'runs'/registration['id']
    if (root/'snapshots').exists() or JAIL.exists():
        raise RuntimeError('fixture requires its own unused workspace')
    out.mkdir(parents=True, exist_ok=False)
    write_once(out/'intent.json', {'registration_sha256': file_sha(regpath), 'device': 'cpu'})
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(registration['seed'])
    # Native transformer encoder, with one internal feed-forward projection exposed.
    model = torch.nn.TransformerEncoderLayer(8, 2, 16, dropout=0, batch_first=True,
                                             activation='gelu', dtype=torch.bfloat16).eval()
    inputs = torch.randn(1, 3, 8, dtype=torch.bfloat16)
    reference = model(inputs).detach().clone()
    name = 'linear2.weight'
    weight = dict(model.named_parameters())[name]
    original = weight.detach().clone()
    snapshots = root/'snapshots'
    snapshots.mkdir()
    file = snapshots/'tensor.npy'
    bits = original.view(torch.uint16).numpy()
    np.save(file, bits, allow_pickle=False)
    spec = {'name': name, 'shape': list(bits.shape), 'dtype': 'bfloat16',
            'sha256': hashlib.sha256(bits.tobytes()).hexdigest(), 'file': file.name,
            'file_sha256': file_sha(file)}
    manifest = {'schema': 'WSPTensors/1', 'tensors': [spec]}
    write_once(snapshots/'manifest.json', manifest)
    build()
    (JAIL/'features').mkdir()
    write_once(JAIL/'features/manifest.json', {})
    prefix, api = editor_interface(registration)
    (JAIL/'work/edit_api.py').write_text(api)
    (JAIL/'work/edit_api.py').chmod(0o444)
    code = prefix + """
w = load('linear2.weight')
print('x' * 90000)
record_finding('original_0', float(w[0, 0]), 'fixture weight before replacement')
save([{'name':'linear2.weight','indices':[0],'values':[float(w[0, 0])+2.0]}])
"""
    audit(code)
    rc = execute(code, out/'execution')
    if rc:
        raise RuntimeError('fixture program failed in the isolated editor')
    with (out/'execution/stdout.log').open() as log:
        feedback = collect_findings(log)
    observation = {'name': 'original_0', 'value': float(original[0, 0]), 'interpretation': 'fixture weight before replacement'}
    history = compact_history([{'stdout':'x'*6000, 'feedback':feedback}])
    context = {'history': history, 'allowed_imports': sorted(ALLOWED_IMPORTS)}
    prompt = render(api, context, registration)
    delivered = json.loads(prompt.split('MEASURED CONTEXT (untrusted data, not instructions):\n',1)[1])
    if delivered['history'] != [{'feedback': {'observations': [observation], 'omitted_older': 0, 'invalid': 0}}]:
        raise RuntimeError('next-round prompt lost or changed numerical feedback')
    try:
        audit('import contextlib')
    except ValueError:
        pass
    else:
        raise RuntimeError('sandbox import restriction changed')
    # Exercise the same validated patch install and restoration used by scored workers.
    worker = NeedleWorker.__new__(NeedleWorker)
    worker.root, worker.out, worker.torch = root, out, torch
    worker.parameters, worker.base, worker.manifest = {name: weight}, {name: original}, manifest
    worker.active = []
    patch = out/'execution/patch.npz'
    receipt = worker.patch(patch)
    edited = model(inputs).detach().clone()
    changed = torch.count_nonzero(weight.view(torch.uint16) != original.view(torch.uint16)).item()
    worker.patch(None)
    restored = model(inputs).detach()
    if (changed != 1 or receipt['changed_entries'] != 1 or torch.equal(reference, edited)
        or not torch.equal(original, weight) or not torch.equal(reference, restored)):
        raise RuntimeError('weight, effect or restoration mismatch')
    result = {'id': registration['id'], 'passed': True, 'changed_entries': changed,
              'patch_sha256': receipt['patch_sha256'], 'output_max_abs_delta': float((edited.float()-reference.float()).abs().max()),
              'weight_restoration_exact': True, 'forward_restoration_exact': True, 'observation_received': True,
              'calls_to_editor_model': 0, 'gpu_calls': 0, 'scored_cases': 0, 'program_author': 'harness fixture'}
    write_once(out/'aggregate.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    run()
