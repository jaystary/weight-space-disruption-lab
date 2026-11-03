"""Post-completion evidence audit only; no model inference or candidate changes."""
import hashlib
import json
from pathlib import Path
import subprocess

from .contracts import canonical, file_sha, write_once
from .needle_contract import (candidate_rank, context_cases, exact_answer, goal_pass,
                              guards, memory_facts, memory_questions)
from .needle_loop import editor_interface, comparison, editor_accounting, trace_deltas
from .tensors import validate_patch

ROOT = Path(__file__).resolve().parents[2]


def identical_pair_inputs(records):
    if len(records) != 2 or records[0]['input_token_sha256'] != records[1]['input_token_sha256']:
        raise ValueError('paired input identity mismatch')


def audit():
    if not str(ROOT).startswith('/workspace/'):
        raise RuntimeError('private evidence stays on remote workspace')
    out = ROOT/'runs/synthetic-needles-01'
    result = {'passed': False, 'inference_calls': 0, 'checks': []}

    def require(condition, label):
        if not condition:
            raise ValueError(label)
        result['checks'].append(label)

    def read(path):
        return json.loads(path.read_text())

    try:
        report, intent = read(out/'aggregate-report.json'), read(out/'intent.json')
        require(report['complete'], 'driver completed both research tracks')
        source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        require(source == intent['source_commit'] == report['source_commit'], 'executed source identity')
        subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'tools', 'tests', 'registrations', 'prompts'],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        regpath = ROOT/'registrations/synthetic-needles-01.json'
        require(file_sha(regpath) == intent['registration_sha256'], 'registration identity')
        reg = read(regpath)
        prefix, _ = editor_interface(reg)
        seed = read(out/'private-seed.json')['seed']
        require(hashlib.sha256(seed.encode()).hexdigest() == read(out/'seed-commitment.json')['sha256'], 'seed commitment')
        frozen = read(out/'choices-frozen.json')
        require(frozen['source_commit'] == source and frozen['registration_sha256'] == file_sha(regpath), 'joint candidate freeze identity')
        require(frozen['seed_commitment'] == hashlib.sha256(seed.encode()).hexdigest(), 'frozen seed identity')
        manifest = read(ROOT/'snapshots/manifest.json')
        require(file_sha(ROOT/'snapshots/manifest.json') == read(out/'worker-ready.json')['manifest_sha256'], 'snapshot manifest identity')
        requests = sorted((out/'model-requests').iterdir())
        require([p.name for p in requests] == [f'{i:04d}' for i in range(1, len(requests)+1)], 'contiguous request ledger')
        generations, tokens, unknown_generations, restored, index, traces = 0, 0, 0, [], {}, {}
        from transformers import AutoTokenizer
        cache = Path('/workspace/weight-space-lab/model-cache/models--Qwen--Qwen3.5-0.8B/snapshots')/reg['revision']
        tokenizer = AutoTokenizer.from_pretrained(cache, local_files_only=True, trust_remote_code=False)
        for path in requests:
            request = read(path/'intent.json')
            require(request['epoch'] <= intent['deadline_epoch'], f'{path.name}: request inside wall cap')
            if request['kind'] == 'generate':
                generations += 1
                if not (path/'private-output.json').exists():
                    parts = request['label']['directory'].split('/')
                    require(len(parts) == 3 and parts[0] in ('context', 'memory') and
                            parts[1].startswith('round-') and parts[2] == 'development' and
                            read(out/parts[0]/parts[1]/'round.json')['status'] == 'failed',
                            f'{path.name}: incomplete request belongs only to a preserved failed development candidate')
                    unknown_generations += 1
                    continue
                raw, receipt = read(path/'private-output.json'), read(path/'input-receipt.json')
                require(tokenizer.decode(raw['tokens'], skip_special_tokens=True) == raw['text'], f'{path.name}: decoded output')
                require(len(raw['tokens']) == raw['row']['tokens'] <= 16, f'{path.name}: token cap')
                tokens += len(raw['tokens'])
                label = request['label']
                key = (label['directory'], label['case'])
                require(key not in index, f'{path.name}: no repeated scored request')
                index[key] = (raw, receipt, request)
            elif request['kind'] == 'restoration':
                restored.append(read(path/'receipt.json')['logits_sha256'])
            else:
                require((path/'trace-receipt.json').exists(), f'{path.name}: trace completed')
                label = request['label']
                trace_key = (label['prefix'], label['case'])
                require(trace_key not in traces, f'{path.name}: no repeated trace')
                traces[trace_key] = read(path/'trace-receipt.json')
        require(len(restored) >= 3 and len(set(restored)) == 1, 'all restoration logits identical')
        require(generations == report['generations'] <= reg['limits']['max_generations'], 'generation count cap')
        require(tokens == report['generated_tokens'] <= reg['limits']['max_generated_tokens'], 'output token accounting')
        require(tokens+16*unknown_generations <= reg['limits']['max_generated_tokens'], 'unknown generation upper bound fits token cap')
        require(len(requests) == report['model_requests'] <= reg['limits']['max_model_requests'], 'request count cap')
        require(report['seconds'] <= reg['limits']['max_wall_seconds'], 'wall limit')
        require(editor_accounting(out) == report['editor_accounting'], 'editor accounting including unknown usage')
        require(report['editor_accounting']['attempts'] <= 8, 'editor call cap')

        def scored_rows(cases, directory, expected_patch, after_freeze=False):
            rows, receipts = [], []
            for case in cases:
                raw, receipt, request = index[(directory, case['id'])]
                rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': case['question']}],
                           tokenize=False, add_generation_prompt=True, enable_thinking=False)
                import numpy as np
                ids = np.asarray([tokenizer(rendered)['input_ids']], dtype=np.int64)
                require(hashlib.sha256(ids.tobytes()).hexdigest() == receipt['input_token_sha256'], 'registered prompt token identity')
                require(receipt['active_patch_sha256'] == expected_patch, 'scored patch identity')
                require(raw['row']['correct'] == (exact_answer(raw['text']) == exact_answer(case['answer'])), 'strict score reconstruction')
                require(raw['row']['id'] == case['id'] and raw['row']['family'] == case['family'], 'case identity')
                if after_freeze:
                    require(request['epoch'] >= frozen['epoch'], 'scored after joint freeze')
                rows.append(raw['row']); receipts.append(receipt)
            return rows, receipts

        development = read(out/'private-development.json')
        facts = memory_facts(seed+'written-facts')
        require(development == {'context': context_cases(seed+'development-context', 16, frozen['selected_context_records']),
                                'memory': memory_questions(facts)}, 'development cohort regeneration')
        for track in ('context', 'memory'):
            chosen = read(out/track/'chosen.json')
            require(chosen == frozen['chosen'][track], f'{track}: selected candidate frozen')
            baseline, _ = scored_rows(development[track], track+'/development-baseline', None)
            best_rank, best_round, best_rows = candidate_rank(baseline, 0), 0, baseline
            for step in range(1, 5):
                directory = out/track/f'round-{step}'
                entry = read(directory/'round.json')
                if entry['status'] != 'evaluated':
                    continue
                patch = directory/'execution/patch.npz'
                _, validation = validate_patch(patch.read_bytes(), manifest, ROOT/'snapshots')
                answer = read(directory/'editor/answer.json')
                require(hashlib.sha256((prefix+answer['code']).encode()).hexdigest() == read(directory/'execution/receipt.json')['executed_code_sha256'], 'executed editor program identity')
                rows, _ = scored_rows(development[track], f'{track}/round-{step}/development', validation['patch_sha256'])
                require(rows == entry['rows'] and comparison(baseline, rows) == entry['comparison'], 'candidate development score')
                rank = candidate_rank(rows, validation['changed_entries'])
                if rank > best_rank:
                    best_rank, best_round, best_rows = rank, step, rows
            require(best_round == chosen['round'] and comparison(baseline, best_rows) == chosen['development'], f'{track}: development-only selection')
            if chosen['patch'] is None:
                require(best_round == 0 and not report['tracks'][track]['final_opened'], f'{track}: no-candidate final remains sealed')
                require(not (out/track/'private-final-cohort.json').exists(), f'{track}: no unused final generated')
                continue
            require(file_sha(Path(chosen['patch'])) == chosen['patch_sha256'], f'{track}: final patch hash')
            diagnostic = out/track/'selected-development-traces'
            trace_manifest = read(diagnostic/'manifest.json')
            for spec in trace_manifest.values():
                require(file_sha(diagnostic/spec['file']) == spec['sha256'], 'selected trace file identity')
            for case in development[track][:4]:
                before, after = traces[('query', case['id'])], traces[('selected', case['id'])]
                identical_pair_inputs([before, after])
                require(before['active_patch_sha256'] is None and after['active_patch_sha256'] == chosen['patch_sha256'], 'selected trace intervention identity')
            require(read(diagnostic/'aggregate-deltas.json')['features'] == trace_deltas(out/track/'features', diagnostic, trace_manifest), f'{track}: development intervention map')
            final = (context_cases(seed+'final-context', 48, frozen['selected_context_records']) if track == 'context'
                     else memory_questions(facts, held_out=True))
            guard = guards(seed+'guard-'+track, include_unknown=track == 'memory')
            require(read(out/track/'private-final-cohort.json') == {'final': final, 'guard': guard}, f'{track}: final regeneration')
            require(read(out/track/'cohort-commitments.json') == {
                key: {'count': len(value), 'sha256': hashlib.sha256(canonical(value)).hexdigest()}
                for key, value in [('final', final), ('guard', guard)]}, f'{track}: final cohort commitments')
            outcomes = {}
            for phase, cases in [('final', final), ('guard', guard)]:
                paired_rows = {'baseline': [], 'edited': []}
                for case in cases:
                    inputs = []
                    for arm in ('baseline', 'edited'):
                        rows, receipts = scored_rows([case], f'{track}/{phase}/{case["id"]}/{arm}',
                                   None if arm == 'baseline' else chosen['patch_sha256'], True)
                        paired_rows[arm].extend(rows); inputs.extend(receipts)
                    identical_pair_inputs(inputs)
                outcomes[phase] = paired_rows
            cmp = comparison(outcomes['final']['baseline'], outcomes['final']['edited'])
            guard_metrics = {f: comparison([r for r in outcomes['guard']['baseline'] if r['family'] == f],
                                           [r for r in outcomes['guard']['edited'] if r['family'] == f])
                             for f in sorted({r['family'] for r in guard})}
            measured = report['tracks'][track]
            require(cmp == measured['comparison'] and guard_metrics == measured['guards'], f'{track}: final paired metrics')
            require(goal_pass(cmp, guard_metrics, True) == measured['goal_met'] and min(1, 4*cmp['one_sided_exact_p']) == measured['goal_adjusted_p'], f'{track}: corrected decision')
        result['passed'] = True
        result['generations'], result['model_requests'] = generations, len(requests)
        result['unknown_development_generations'] = unknown_generations
    except Exception as error:
        result['failure'] = {'type': type(error).__name__, 'message': str(error)[:500]}
    write_once(out/'closure-checks.json', result['checks'])
    result['check_count'] = len(result.pop('checks'))
    write_once(out/'closure-check.json', result)
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    audit()
