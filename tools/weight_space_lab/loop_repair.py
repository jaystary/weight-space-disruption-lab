"""Remote-only, one-shot evaluation of the authors' exact BF16 loop-repair patch."""
import hashlib
import json
from pathlib import Path
import secrets
import subprocess
import time
import urllib.request

from .contracts import canonical, file_sha, write_once
from .loop_contract import arithmetic, boxed_integer, decision, paired, repeated_fraction

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = ROOT/'registrations/loop-repair-03.json'


def verified_patch(model, registration, raw):
    import io
    import numpy as np
    import torch
    expected = registration['upstream_patch_git_blob_sha1']
    blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if len(raw) != registration['upstream_patch_bytes'] or blob != expected:
        raise ValueError('published patch differs from pinned Git object')
    parameters = dict(model.named_parameters())
    originals, edits = {}, {}
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if set(archive.files) != {'t0_idx', 't0_val', 't1_idx', 't1_val'}:
            raise ValueError('unexpected published sparse patch members')
        for i, spec in enumerate(registration['tensors']):
            p = parameters[spec['name']]
            if p.dtype != torch.bfloat16 or list(p.shape) != spec['shape']:
                raise ValueError('base shape or dtype mismatch')
            bits = p.detach().cpu().contiguous().view(torch.uint16).numpy().copy()
            if hashlib.sha256(bits.tobytes()).hexdigest() != spec['base_sha256']:
                raise ValueError('base tensor does not match original publication')
            indices, values = archive[f't{i}_idx'], archive[f't{i}_val']
            if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or values.dtype != np.uint16
                    or values.shape != indices.shape or len(indices) != spec['changed_entries']
                    or np.any(indices < 0) or np.any(indices >= bits.size)
                    or len(np.unique(indices)) != len(indices)):
                raise ValueError('invalid sparse patch')
            changed = bits.copy()
            changed.reshape(-1)[indices] = values
            if (int(np.count_nonzero(changed != bits)) != spec['changed_entries']
                    or hashlib.sha256(changed.tobytes()).hexdigest() != spec['edited_sha256']):
                raise ValueError('edited tensor does not match original publication')
            originals[spec['name']] = torch.from_numpy(bits).view(torch.bfloat16)
            edits[spec['name']] = torch.from_numpy(changed).view(torch.bfloat16)
    return parameters, originals, edits


def run():
    if not str(ROOT).startswith('/workspace/'):
        raise RuntimeError('experiment may run only on the authorized remote workspace')
    import torch
    import numpy as np
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, LogitsProcessor, LogitsProcessorList

    if not torch.cuda.is_available():
        raise RuntimeError('authorized remote GPU is required')
    registration = json.loads(REGISTRATION.read_text())
    out = ROOT/'runs/loop-repair-03'
    out.mkdir(parents=True, exist_ok=False)
    out.chmod(0o700)
    start = time.monotonic()
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'tools', 'registrations', 'tests'], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    write_once(out/'intent.json', {'source_commit': source, 'registration_sha256': file_sha(REGISTRATION),
                                 'started_epoch': time.time(), 'limits': registration['limits']})
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.cuda.set_per_process_memory_fraction(.9)
    rows, errors = [], []
    attempts = 0
    generated = 0

    def within_budget():
        if time.monotonic()-start >= registration['limits']['max_wall_seconds']:
            raise RuntimeError('registered wall-clock limit reached')
        if attempts >= registration['limits']['max_generation_attempts']:
            raise RuntimeError('registered generation-attempt limit reached')
        if generated >= registration['limits']['max_generated_tokens']:
            raise RuntimeError('registered generated-token limit reached')

    try:
        model_path = snapshot_download(registration['model'], revision=registration['revision'],
                                       cache_dir='/workspace/weight-space-lab/model-cache',
                                       allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt'])
        patch_url = ('https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/'
                     +registration['upstream_commit']+'/'+registration['upstream_patch_path'])
        with urllib.request.urlopen(patch_url, timeout=60) as response:
            raw = response.read(registration['upstream_patch_bytes']+1)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map={'': 0}, local_files_only=True,
            trust_remote_code=False, attn_implementation='eager').eval()
        parameters, original, edited = verified_patch(model, registration, raw)
        (out/'published-edit.npz').write_bytes(raw)
        write_once(out/'weight-verification.json', {'all_base_and_edited_hashes_verified': True,
                   'changed_entries': sum(s['changed_entries'] for s in registration['tensors']),
                   'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)})

        # This fixes every choice before final questions are generated or opened.
        seed = secrets.token_hex(32)
        write_once(out/'private-seed.json', {'seed': seed})
        write_once(out/'choices-frozen.json', {'source_commit': source, 'registration_sha256': file_sha(REGISTRATION),
                   'seed_commitment': hashlib.sha256(seed.encode()).hexdigest(),
                   'patch_sha256': hashlib.sha256(raw).hexdigest(), 'selection': 'none'})
        used = set()
        phases = {
            'development': arithmetic(seed+'development', registration['data']['development_count'], used),
            'final': arithmetic(seed+'final', registration['data']['final_count'], used),
            'guard': [{'id': f'guard-{i}', 'family': 'exact-string',
                       'answer': hashlib.sha256((seed+f'guard-{i}').encode()).hexdigest()[:8]}
                      for i in range(registration['data']['guard_count'])]}
        write_once(out/'private-cohorts.json', phases)
        write_once(out/'cohort-commitments.json', {k: {'count': len(v), 'sha256': hashlib.sha256(canonical(v)).hexdigest()}
                                                for k, v in phases.items()})

        class PresencePenalty(LogitsProcessor):
            def __init__(self, prompt_length):
                self.prompt_length = prompt_length

            def __call__(self, input_ids, scores):
                for i in range(input_ids.shape[0]):
                    seen = input_ids[i, self.prompt_length:].unique()
                    scores[i, seen] -= registration['generation']['presence_penalty']
                return scores

        # A synthetic, unscored restoration check catches state carry-over.
        probe = tokenizer('2+2=', return_tensors='pt').to('cuda')
        with torch.inference_mode():
            reference = model(**probe).logits[:, -1].clone()
            for name, value in edited.items():
                parameters[name].copy_(value)
            for name, value in original.items():
                parameters[name].copy_(value)
            restored = model(**probe).logits[:, -1]
        if not torch.equal(reference, restored):
            raise RuntimeError('identity restoration logits differ')
        write_once(out/'restoration-check.json', {'exact_logits_equal': True, 'synthetic_forwards': 2})

        for phase, cohort in phases.items():
            for case_index, case in enumerate(cohort):
                question = (f'Return exactly this string and nothing else: {case["answer"]}' if phase == 'guard'
                            else case['question']+' Please reason step by step, and put your final answer within \\boxed{}.')
                rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False,
                                                          add_generation_prompt=True, enable_thinking=True)
                inputs = tokenizer(rendered, return_tensors='pt').to('cuda')
                prompt_length = inputs.input_ids.shape[1]
                if prompt_length > 1024:
                    raise RuntimeError('authored prompt exceeds fixed input bound')
                sampling_seed = int(hashlib.sha256((seed+case['id']).encode()).hexdigest()[:8], 16)
                arms = registration['arms'] if case_index % 2 == 0 else list(reversed(registration['arms']))
                for arm in arms:
                    within_budget()
                    attempt = out/phase/case['id']/arm
                    attempt.mkdir(parents=True, exist_ok=False)
                    write_once(attempt/'intent.json', {'case': case['id'], 'arm': arm, 'sampling_seed': sampling_seed})
                    attempts += 1
                    for name, value in (original if arm == 'baseline' else edited).items():
                        parameters[name].copy_(value)
                    torch.manual_seed(sampling_seed)
                    torch.cuda.manual_seed_all(sampling_seed)
                    cap = registration['generation']['guard_max_new_tokens' if phase == 'guard' else 'max_new_tokens']
                    torch.cuda.reset_peak_memory_stats()
                    begun = time.monotonic()
                    with torch.inference_mode():
                        result = model.generate(**inputs, do_sample=True, temperature=.6, top_p=.95, top_k=20,
                            repetition_penalty=1.0, max_new_tokens=cap, use_cache=True,
                            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                            logits_processor=LogitsProcessorList([PresencePenalty(prompt_length)]))
                    torch.cuda.synchronize()
                    tokens = result[0, prompt_length:].tolist()
                    text = tokenizer.decode(tokens, skip_special_tokens=False)
                    answer_text = text.split('</think>')[-1]
                    for special in tokenizer.all_special_tokens:
                        answer_text = answer_text.replace(special, '')
                    prediction = answer_text.strip() if phase == 'guard' else boxed_integer(answer_text)
                    generated += len(tokens)
                    row = {'phase': phase, 'id': case['id'], 'family': case['family'], 'arm': arm,
                           'correct': prediction == case['answer'], 'prediction': prediction,
                           'completed': tokens[-1] == tokenizer.eos_token_id,
                           'answer_present': prediction is not None and prediction != '',
                           'tokens': len(tokens), 'hit_token_cap': len(tokens) == cap and tokens[-1] != tokenizer.eos_token_id,
                           'repeated_8gram_fraction': repeated_fraction(tokens),
                           'seconds': time.monotonic()-begun, 'peak_gpu_bytes': torch.cuda.max_memory_allocated()}
                    write_once(attempt/'private-output.json', {'text': text, 'tokens': tokens})
                    write_once(attempt/'score.json', row)
                    rows.append(row)
                print(json.dumps({'phase': phase, 'completed_pairs': case_index+1,
                                  'generated_tokens': generated, 'elapsed_seconds': time.monotonic()-start}), flush=True)
        for name, value in original.items():
            parameters[name].copy_(value)
    except Exception as error:
        errors.append({'type': type(error).__name__, 'message': str(error)[:500]})
        write_once(out/'failure.json', errors[-1])
    complete = len(rows) == registration['limits']['max_generation_attempts'] and not errors
    comparisons = {}
    for phase in ('development', 'final', 'guard'):
        selected = [r for r in rows if r['phase'] == phase]
        indexed = {(r['id'], r['arm']): r for r in selected}
        ids = sorted({r['id'] for r in selected if all((r['id'], a) in indexed for a in registration['arms'])})
        if ids:
            comparisons[phase] = paired([int(indexed[(i, 'baseline')]['correct']) for i in ids],
                                        [int(indexed[(i, 'published-p50c')]['correct']) for i in ids])
    aggregates = {}
    for phase in ('development', 'final', 'guard'):
        aggregates[phase] = {}
        for arm in registration['arms']:
            selected = [r for r in rows if r['phase'] == phase and r['arm'] == arm]
            if selected:
                aggregates[phase][arm] = {key: float(np.mean([r[key] for r in selected]))
                    for key in ('correct', 'tokens', 'completed', 'answer_present', 'hit_token_cap', 'repeated_8gram_fraction', 'seconds')}
                aggregates[phase][arm]['count'] = len(selected)
                aggregates[phase][arm]['peak_gpu_bytes'] = max(r['peak_gpu_bytes'] for r in selected)
                aggregates[phase][arm]['by_family'] = {str(f): {'count': sum(r['family'] == f for r in selected),
                    'accuracy': float(np.mean([r['correct'] for r in selected if r['family'] == f]))}
                    for f in sorted({r['family'] for r in selected})}
    status = decision(comparisons.get('final', {}), comparisons.get('guard', {}), complete, len(errors))
    report = {'id': registration['id'], 'source_commit': source, 'status': status, 'complete': complete,
              'comparisons': comparisons, 'aggregates': aggregates, 'errors': errors,
              'attempts': attempts, 'generated_tokens': generated, 'seconds': time.monotonic()-start,
              'interpretation': registration['purpose'], 'differences_from_original': registration['differences_from_original']}
    write_once(out/'aggregate-report.json', report)
    print(json.dumps(report), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    run()
