"""Fresh, bounded longer-context replication; never resumes or changes trial03."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import secrets
import signal
import subprocess
import time

from .contracts import canonical, file_sha, write_once
from .loop_contract import arithmetic, boxed_integer, paired, repeated_fraction
from .loop_repair import verified_patch

ROOT = Path(__file__).resolve().parents[2]
PRIOR = Path('/workspace/weight-space-loop-03')


def trim_generated(tokens, eos):
    for i, token in enumerate(tokens):
        if token in eos:
            return tokens[:i+1], True
    return tokens, False


def fresh_arithmetic(seed, count, forbidden):
    for attempt in range(100):
        rows = arithmetic(seed+str(attempt), count)
        ids = {row['id'] for row in rows}
        if ids.isdisjoint(forbidden):
            forbidden.update(ids)
            return rows
    raise RuntimeError('identity-only collision rejection exhausted')


def meets_goal(result, guards, complete, errors):
    return bool(complete and not errors and result['gain'] >= .10
                and min(1.0, 2*result['one_sided_exact_p']) <= .05 and guards['gain'] >= 0)


def run():
    if not str(ROOT).startswith('/workspace/'):
        raise RuntimeError('remote workspace required')
    if not (PRIOR/'provenance/loop-repair-03-driver-exit.txt').exists():
        raise RuntimeError('prior driver must be terminal')
    prior_report = json.loads((PRIOR/'runs/loop-repair-03/aggregate-report.json').read_text())
    if not json.loads((PRIOR/'runs/loop-repair-03/closure-check.json').read_text())['passed']:
        raise RuntimeError('prior closure audit must pass')
    if meets_goal(prior_report['comparisons']['final'], prior_report['comparisons']['guard'],
                  prior_report['complete'], prior_report['errors']):
        print(json.dumps({'state': 'not_started', 'reason': 'prior_trial_meets_goal_family'}))
        return
    import torch
    import numpy as np
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, LogitsProcessor, LogitsProcessorList

    if not torch.cuda.is_available():
        raise RuntimeError('authorized GPU required')
    reg_path = ROOT/'registrations/loop-repair-04.json'
    reg = json.loads(reg_path.read_text())
    hard_stop = datetime.datetime.fromisoformat(reg['admission']['hard_stop_utc'].replace('Z', '+00:00')).timestamp()
    if time.time()+reg['limits']['max_wall_seconds']+reg['admission']['closure_reserve_seconds'] > hard_stop:
        raise RuntimeError('full registered wall allowance no longer fits before closure reserve')
    out = ROOT/'runs/loop-repair-04'
    out.mkdir(parents=True, exist_ok=False)
    out.chmod(0o700)
    start = time.monotonic()
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'tools', 'registrations', 'tests'], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    write_once(out/'intent.json', {'source_commit': source, 'registration_sha256': file_sha(reg_path),
               'goal_family_sha256': file_sha(ROOT/'registrations/weight-only-goal-01.json'),
               'started_epoch': time.time(), 'limits': reg['limits']})
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.cuda.set_per_process_memory_fraction(.9)
    rows, errors = [], []
    attempts = generated = token_slots = 0
    cfg = reg['generation']
    eos = cfg['eos_token_ids']

    def budget():
        if time.monotonic()-start >= reg['limits']['max_wall_seconds'] or time.time() >= hard_stop-reg['admission']['closure_reserve_seconds']:
            raise RuntimeError('registered time budget reached')
        if attempts >= reg['limits']['max_generation_attempts'] or generated >= reg['limits']['max_generated_tokens']:
            raise RuntimeError('registered generation budget reached')

    try:
        path = snapshot_download(reg['model'], revision=reg['revision'],
                                 cache_dir='/workspace/weight-space-lab/model-cache', local_files_only=True,
                                 allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt'])
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False, padding_side='left')
        if tokenizer.eos_token_id != eos[0]:
            raise ValueError('unexpected tokenizer EOS identity')
        tokenizer.pad_token_id = eos[0]
        model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16,
            device_map={'': 0}, local_files_only=True, trust_remote_code=False, attn_implementation='sdpa').eval()
        if model.config.text_config.eos_token_id != eos[1]:
            raise ValueError('unexpected native EOS identity')
        raw = (PRIOR/'runs/loop-repair-03/published-edit.npz').read_bytes()
        parameters, original, edited = verified_patch(model, reg, raw)
        (out/'published-edit.npz').write_bytes(raw)
        write_once(out/'weight-verification.json', {'all_base_and_edited_hashes_verified': True,
            'changed_entries': sum(t['changed_entries'] for t in reg['tensors']), 'torch': torch.__version__})
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

        # Full-context, unscored throughput avoids admission based on short prompts.
        def preflight_timeout(signum, frame):
            raise TimeoutError('registered preflight wall limit reached')
        signal.signal(signal.SIGALRM, preflight_timeout)
        signal.alarm(reg['limits']['max_preflight_seconds'])
        preflight_start = time.monotonic()
        ids = tokenizer(' red blue'*16384, add_special_tokens=False)['input_ids'][:16384]
        if len(ids) != 16384:
            raise RuntimeError('synthetic timing context length mismatch')
        probe_ids = torch.tensor([ids]*cfg['batch_size'], device='cuda')
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(0)
        with torch.inference_mode():
            timed = model.generate(input_ids=probe_ids, attention_mask=torch.ones_like(probe_ids),
                max_new_tokens=128, min_new_tokens=128, do_sample=False, use_cache=True,
                pad_token_id=eos[0], eos_token_id=eos)
        torch.cuda.synchronize()
        signal.alarm(0)
        seconds = time.monotonic()-preflight_start
        expected = 128*cfg['batch_size']
        if timed.shape[1]-probe_ids.shape[1] != 128 or expected > reg['limits']['max_preflight_generated_tokens']:
            raise RuntimeError('preflight output bound mismatch')
        batches = math.ceil(reg['data']['development_count']/8)+math.ceil(reg['data']['final_count']/8)
        steps = 2*batches*cfg['max_new_tokens']+2*math.ceil(reg['data']['guard_count']/8)*cfg['guard_max_new_tokens']
        estimate = steps/(128/seconds)+300
        remaining = reg['limits']['max_wall_seconds']-(time.monotonic()-start)
        admitted = seconds <= reg['limits']['max_preflight_seconds'] and estimate <= .9*remaining and time.time()+estimate < hard_stop-reg['admission']['closure_reserve_seconds']
        write_once(out/'batch-preflight.json', {'seconds': seconds, 'generated_tokens': expected,
            'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'estimated_remaining_seconds': estimate,
            'admitted': admitted, 'efficacy_scored': False})
        del timed, probe_ids, probe, reference, restored
        torch.cuda.empty_cache()
        if not admitted:
            raise RuntimeError('long-context timing did not admit a complete bounded trial')

        seed = secrets.token_hex(32)
        write_once(out/'private-seed.json', {'seed': seed})
        forbidden = {p.name for phase in ('development', 'final') for p in (PRIOR/'runs/loop-repair-03'/phase).iterdir() if p.is_dir()}
        write_once(out/'excluded-ids.json', sorted(forbidden))
        write_once(out/'choices-frozen.json', {'source_commit': source, 'registration_sha256': file_sha(reg_path),
            'goal_family_sha256': file_sha(ROOT/'registrations/weight-only-goal-01.json'),
            'seed_commitment': hashlib.sha256(seed.encode()).hexdigest(),
            'excluded_ids_sha256': file_sha(out/'excluded-ids.json'), 'patch_sha256': hashlib.sha256(raw).hexdigest(), 'selection': 'none'})
        phases = {k: fresh_arithmetic(seed+k, reg['data'][k+'_count'], forbidden) for k in ('development', 'final')}
        phases['guard'] = [{'id': f'guard-{i}', 'family': 'exact-string',
            'answer': hashlib.sha256((seed+f'guard-{i}').encode()).hexdigest()[:8]} for i in range(reg['data']['guard_count'])]
        write_once(out/'private-cohorts.json', phases)
        write_once(out/'cohort-commitments.json', {k: {'count': len(v), 'sha256': hashlib.sha256(canonical(v)).hexdigest()} for k,v in phases.items()})

        class PresencePenalty(LogitsProcessor):
            def __init__(self, prompt_length):
                self.prompt_length = prompt_length
            def __call__(self, input_ids, scores):
                for i in range(input_ids.shape[0]):
                    scores[i, input_ids[i, self.prompt_length:].unique()] -= cfg['presence_penalty']
                return scores

        for phase, cohort in phases.items():
            for batch_number, offset in enumerate(range(0, len(cohort), cfg['batch_size'])):
                batch = cohort[offset:offset+cfg['batch_size']]
                prompts = []
                for case in batch:
                    question = (f'Return exactly this string and nothing else: {case["answer"]}' if phase == 'guard'
                        else case['question']+' Please reason step by step, and put your final answer within \\boxed{}.')
                    prompts.append(tokenizer.apply_chat_template([{'role':'user','content':question}], tokenize=False,
                        add_generation_prompt=True, enable_thinking=cfg['guard_enable_thinking'] if phase == 'guard' else cfg['enable_thinking']))
                inputs = tokenizer(prompts, return_tensors='pt', padding=True).to('cuda')
                length = inputs.input_ids.shape[1]
                if length > 1024:
                    raise RuntimeError('authored prompt exceeds input cap')
                sampling_seed = int(hashlib.sha256((seed+phase+str(batch_number)).encode()).hexdigest()[:8], 16)
                arms = reg['arms'] if batch_number % 2 == 0 else list(reversed(reg['arms']))
                for arm in arms:
                    budget()
                    for case in batch:
                        attempt = out/phase/case['id']/arm
                        attempt.mkdir(parents=True, exist_ok=False)
                        write_once(attempt/'intent.json', {'case':case['id'], 'arm':arm, 'batch':batch_number, 'sampling_seed':sampling_seed})
                    attempts += len(batch)
                    for name, value in (original if arm == 'baseline' else edited).items():
                        parameters[name].copy_(value)
                    torch.manual_seed(sampling_seed)
                    torch.cuda.manual_seed_all(sampling_seed)
                    cap = cfg['guard_max_new_tokens' if phase == 'guard' else 'max_new_tokens']
                    begun = time.monotonic()
                    torch.cuda.reset_peak_memory_stats()
                    with torch.inference_mode():
                        result = model.generate(**inputs, do_sample=cfg['do_sample'], temperature=cfg['temperature'],
                            top_p=cfg['top_p'], top_k=cfg['top_k'], repetition_penalty=cfg['repetition_penalty'],
                            max_new_tokens=cap, use_cache=True, pad_token_id=eos[0], eos_token_id=eos,
                            logits_processor=LogitsProcessorList([PresencePenalty(length)]))
                    torch.cuda.synchronize()
                    elapsed = time.monotonic()-begun
                    token_slots += len(batch)*(result.shape[1]-length)
                    if token_slots > reg['limits']['max_generated_tokens']:
                        raise RuntimeError('registered padded-token compute bound exceeded')
                    for i,case in enumerate(batch):
                        tokens, completed = trim_generated(result[i, length:].tolist(), eos)
                        text = tokenizer.decode(tokens, skip_special_tokens=False)
                        answer = text.split('</think>')[-1]
                        for special in tokenizer.all_special_tokens:
                            answer = answer.replace(special, '')
                        prediction = answer.strip() if phase == 'guard' else boxed_integer(answer)
                        generated += len(tokens)
                        row = {'phase':phase, 'id':case['id'], 'family':case['family'], 'arm':arm,
                            'correct':prediction == case['answer'], 'prediction':prediction, 'tokens':len(tokens),
                            'completed':completed, 'answer_present':prediction is not None and prediction != '',
                            'hit_token_cap':len(tokens)==cap and not completed, 'repeated_8gram_fraction':repeated_fraction(tokens),
                            'batch_seconds':elapsed, 'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
                        attempt = out/phase/case['id']/arm
                        write_once(attempt/'private-output.json', {'text':text,'tokens':tokens})
                        write_once(attempt/'score.json', row)
                        rows.append(row)
                print(json.dumps({'phase':phase,'completed_pairs':offset+len(batch),'generated_tokens':generated,'padded_token_slots':token_slots,'elapsed_seconds':time.monotonic()-start}), flush=True)
        for name,value in original.items():
            parameters[name].copy_(value)
    except Exception as error:
        signal.alarm(0)
        errors.append({'type':type(error).__name__, 'message':str(error)[:500]})
        write_once(out/'failure.json', errors[-1])
    complete = len(rows)==reg['limits']['max_generation_attempts'] and not errors
    comparisons, aggregates = {}, {}
    for phase in ('development','final','guard'):
        selected = [r for r in rows if r['phase']==phase]
        index = {(r['id'],r['arm']):r for r in selected}
        ids = sorted({r['id'] for r in selected if all((r['id'],a) in index for a in reg['arms'])})
        if ids:
            comparisons[phase] = paired([int(index[i,'baseline']['correct']) for i in ids], [int(index[i,'published-p50c']['correct']) for i in ids])
            comparisons[phase]['goal_family_adjusted_p'] = min(1.,2*comparisons[phase]['one_sided_exact_p'])
        aggregates[phase] = {}
        for arm in reg['arms']:
            subset = [r for r in selected if r['arm']==arm]
            if subset:
                aggregates[phase][arm] = {k:float(np.mean([r[k] for r in subset])) for k in ('correct','tokens','completed','answer_present','hit_token_cap','repeated_8gram_fraction')}
                aggregates[phase][arm]['count'] = len(subset)
    success = meets_goal(comparisons.get('final',{}), comparisons.get('guard',{}), complete, errors)
    report = {'id':reg['id'], 'source_commit':source, 'complete':complete, 'status':'goal_bar_met' if success else 'goal_bar_not_met' if complete else 'incomplete_no_success_claim',
        'comparisons':comparisons, 'aggregates':aggregates, 'errors':errors, 'attempts':attempts,
        'generated_tokens':generated, 'padded_token_slots':token_slots, 'seconds':time.monotonic()-start, 'interpretation':reg['purpose'], 'differences_from_trial03':reg['differences_from_trial03']}
    write_once(out/'aggregate-report.json', report)
    print(json.dumps(report), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    run()
