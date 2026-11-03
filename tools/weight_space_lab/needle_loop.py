"""Two bounded synthetic direct-edit tracks; both candidates freeze before finals."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import secrets
import shutil
import subprocess
import time

import numpy as np

from .contracts import canonical, file_sha, write_once
from .edit_loop import ALLOWED_IMPORTS, audit, SCHEMA
from .loop_contract import paired
from .model_client import call
from .needle_contract import (candidate_rank, choose_difficulty, context_cases,
                              goal_pass, guards, memory_facts, memory_questions)
from .needle_findings import collect_findings
from .needle_worker import NeedleWorker
from .research_prompt import render
from .sandbox import build, execute, JAIL

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'import numpy as np\nimport json, math, statistics, collections, hashlib\nfrom edit_api import SPECS, load, save, feature\n'


def editor_interface(registration):
    modules = ['edit_api.py', 'needle_feature_api.py']
    prefix = PREFIX
    mode = registration.get('editor_feedback')
    if mode == 'structured-v1':
        modules.append('needle_findings.py')
        prefix += 'from edit_api import record_finding\n'
    elif mode is not None:
        raise ValueError('unknown editor feedback protocol')
    api = '\n'.join((ROOT/'tools/weight_space_lab'/name).read_text() for name in modules)
    return prefix, api


class RestorationError(RuntimeError):
    pass


def editor_accounting(out):
    attempts, receipts = [], []
    for track in ('context', 'memory'):
        for step in range(1, 5):
            path = out/track/f'round-{step}'/'editor'
            if (path/'intent.json').exists():
                attempts.append(f'{track}/{step}')
                receipt = json.loads((path/'receipt.json').read_text()) if (path/'receipt.json').exists() else {}
                receipts.append({'attempt': f'{track}/{step}', 'usage_known': receipt.get('usage_known', False),
                                 'returncode': receipt.get('returncode'), 'error': receipt.get('error'),
                                 'requested_model': receipt.get('model'), 'reported_models': receipt.get('reported_models', []),
                                 'usage': receipt.get('usage', [])})
    return {'attempts': len(attempts), 'unknown_usage_calls': sum(not r['usage_known'] for r in receipts),
            'receipts': receipts}


def compact_development(cases):
    result = []
    for case in cases:
        row = dict(case)
        question = case['question']
        row['full_question_sha256'] = hashlib.sha256(question.encode()).hexdigest()
        row['question_complete'] = len(question) <= 1200
        if not row['question_complete']:
            row['question'] = question[:200]+'\n[Middle of synthetic table omitted here; evaluator uses the complete input.]\n'+question[-300:]
        result.append(row)
    return result


def compact_history(history):
    result = []
    for i, entry in enumerate(history):
        row = dict(entry)
        if i != len(history)-1:
            row.pop('code', None)
        for key, cap in [('stdout', 2000), ('stderr', 800), ('hypothesis', 2000)]:
            if key in row:
                row[key] = row[key][:cap]
        if row.get('feedback', {}).get('observations'):
            row.pop('stdout', None)
        result.append(row)
    return result


def trace_deltas(baseline_directory, selected_directory, inventory):
    baseline = json.loads((baseline_directory/'manifest.json').read_text())
    result = {}
    for name, spec in inventory.items():
        original_name = name.replace('selected/', 'query/', 1)
        after = np.load(selected_directory/spec['file'], allow_pickle=False)
        before = np.load(baseline_directory/baseline[original_name]['file'], allow_pickle=False)[:len(after)]
        if after.ndim != 2 or len(after) != 4 or after.shape != before.shape or not np.isfinite(after).all() or not np.isfinite(before).all():
            raise ValueError('invalid paired development trace')
        delta = after-before
        measured = {'cases': len(after), 'mean_delta_l2': float(np.linalg.norm(delta, axis=-1).mean()),
                    'max_abs_delta': float(np.max(np.abs(delta)))}
        if after.shape[-1] <= 32:
            measured.update(mean_before=before.mean(axis=0).tolist(), mean_after=after.mean(axis=0).tolist())
        result[original_name] = measured
    return result


def comparison(base, edited):
    if [r['id'] for r in base] != [r['id'] for r in edited]:
        raise ValueError('unaligned comparisons')
    return paired([int(r['correct']) for r in base], [int(r['correct']) for r in edited])


def install_features(worker, cases, track, out):
    directory = out/'features'
    directory.mkdir()
    inventory = worker.traces(cases, 'query', directory)
    if track == 'memory':
        supported = [dict(r, question=f'Known association: {r["entity"]} has designated color {r["answer"]}.\n\n'+r['question'],
                          focus_text=f'{r["entity"]} has designated color {r["answer"]}.') for r in cases]
        inventory.update(worker.traces(supported, 'supported', directory))
    ids = [worker.tokenizer.encode(r['answer'], add_special_tokens=False)[0] for r in cases]
    head = worker.model.get_output_embeddings().weight[ids].detach().float().cpu().numpy()
    path = directory/'answer-directions.npy'
    np.save(path, head, allow_pickle=False)
    inventory['answer_first_token_head'] = {'file': path.name, 'shape': list(head.shape), 'sha256': file_sha(path)}
    write_once(directory/'manifest.json', inventory)
    target = JAIL/'features'
    target.mkdir(exist_ok=True)
    # Each track has a closed allowlist; prior features are never exposed by name.
    for spec in inventory.values():
        target_file = target/spec['file']
        if target_file.exists():
            target_file.chmod(0o644)
        shutil.copyfile(directory/spec['file'], target_file)
        target_file.chmod(0o444)
    manifest = target/'manifest.json'
    if manifest.exists():
        manifest.chmod(0o644)
    manifest.write_bytes(canonical(inventory))
    manifest.chmod(0o444)
    return inventory


def edit_track(worker, cases, track, out, registration, reference, deadline):
    out.mkdir()
    worker.patch(None)
    baseline = worker.generate(cases, out/'development-baseline')
    features = install_features(worker, cases, track, out)
    prefix, api = editor_interface(registration)
    target = JAIL/'work/edit_api.py'
    if target.exists():
        target.chmod(0o644)
    target.write_text(api)
    target.chmod(0o444)
    best = {'patch': None, 'changed_entries': 0, 'round': 0, 'rows': baseline}
    history = []
    prior = Path('/workspace/weight-space-needle-runtime-bridge-01/runs/needle-runtime-bridge-01/execution/stdout.log')
    numerical_notes = ('Partial prior output: analysis failed on missing hashlib binding; no complete numerical conclusion.\n'
                       +prior.read_text()[:12000]) if prior.exists() else 'Prior numerical inspection unavailable.'
    for step in range(1, 5):
        directory = out/f'round-{step}'
        directory.mkdir()
        entry = {'round': step, 'status': 'failed'}
        try:
            if time.time()+registration['limits']['editor_timeout_seconds']+120 >= deadline:
                raise RuntimeError('insufficient reserved time for another editor round')
            context = {'track': track, 'round': step, 'remaining_calls_including_this': 5-step,
                       'immutable_tensors': worker.manifest,
                       'features': {name: {'shape': spec['shape']} for name, spec in features.items()},
                       'development': compact_development(cases), 'baseline': baseline, 'best': best['rows'],
                       'history': compact_history(history), 'numerical_weight_inspection': numerical_notes,
                       'kernel_bindings': prefix,
                       'response': 'Return hypothesis and executable code strings only. Code executes once. Print compact diagnostics; optionally save a sparse internal-weight patch relative to BASE. The controller evaluates each valid patch on this fixed development panel. A diagnostic-only round consumes one call. No direct GPU function exists inside your program.'}
            if registration.get('editor_feedback') == 'structured-v1':
                context['allowed_imports'] = sorted(ALLOWED_IMPORTS)
                context['diagnostic_feedback'] = (
                    'Use bound record_finding(name, float(value), interpretation). The last four '
                    'valid observations reach the next round separately from printed logs. '
                    'Use bound save(changes) for writes. The controller verifies and hashes the patch.')
            prompt = render(api, context, registration)
            if len(prompt) > registration['limits']['editor_prompt_characters']:
                raise ValueError('editor prompt budget exceeded')
            answer, receipt = call(prompt, SCHEMA, directory/'editor', model=registration['editor_model'],
                                   effort=registration['editor_effort'], timeout=480,
                                   executable='/workspace/tools/codex')
            code = prefix+answer['code']
            write_once(directory/'code-audit.json', audit(code))
            rc = execute(code, directory/'execution')
            entry.update(hypothesis=answer['hypothesis'], code=answer['code'],
                         stdout=(directory/'execution/stdout.log').read_text(errors='replace')[:6000],
                         stderr=(directory/'execution/stderr.log').read_text(errors='replace')[:1500],
                         program_returncode=rc, usage_known=receipt['usage_known'])
            if registration.get('editor_feedback') == 'structured-v1':
                with (directory/'execution/stdout.log').open(encoding='utf-8', errors='replace') as stream:
                    entry['feedback'] = collect_findings(stream)
            if rc:
                raise RuntimeError('generated program failed')
            patch = directory/'execution/patch.npz'
            if not patch.exists():
                entry['status'] = 'diagnostic_only'
            else:
                validation = worker.patch(patch)
                write_once(directory/'patch-validation.json', validation)
                rows = worker.generate(cases, directory/'development')
                worker.patch(None)
                try:
                    restored = worker.restoration_probe({'track': track, 'round': step})
                except Exception as error:
                    raise RestorationError('restoration probe failed; editing stopped') from error
                if not worker.torch.equal(reference, restored):
                    raise RestorationError('base restoration logits differ; editing stopped')
                entry.update(status='evaluated', comparison=comparison(baseline, rows),
                             rows=rows, changed_entries=validation['changed_entries'])
                if candidate_rank(rows, validation['changed_entries']) > candidate_rank(best['rows'], best['changed_entries']):
                    best = {'patch': str(patch), 'changed_entries': validation['changed_entries'],
                            'round': step, 'rows': rows}
        except RestorationError as error:
            worker.patch(None)
            entry['error'] = str(error)
            write_once(directory/'round.json', entry)
            raise
        except Exception as error:
            worker.patch(None)
            entry['error'] = type(error).__name__+': '+str(error)[:500]
        write_once(directory/'round.json', entry)
        history.append(entry)
        print(json.dumps({'phase': 'editing', 'track': track, 'round': step, 'status': entry['status'],
                          'comparison': entry.get('comparison'), 'error': entry.get('error')}), flush=True)
    worker.patch(None)
    chosen = {'patch': best['patch'], 'patch_sha256': file_sha(Path(best['patch'])) if best['patch'] else None,
              'changed_entries': best['changed_entries'], 'round': best['round'],
              'development': comparison(baseline, best['rows']),
              'round_outcomes': [r['status'] for r in history], 'uses_final': False}
    write_once(out/'chosen.json', chosen)
    return chosen


def final_pairs(worker, cases, patch, directory):
    directory.mkdir(parents=True, exist_ok=False)
    rows = {'baseline': [], 'edited': []}
    for i, case in enumerate(cases):
        for arm in (('baseline', 'edited') if i % 2 == 0 else ('edited', 'baseline')):
            worker.patch(None if arm == 'baseline' else Path(patch))
            measured = worker.generate([case], directory/case['id']/arm)
            rows[arm].extend(measured)
        print(json.dumps({'phase': str(directory.relative_to(worker.out)), 'completed_pairs': i+1,
                          'generated_tokens': worker.tokens}), flush=True)
    worker.patch(None)
    return rows


def run():
    if not str(ROOT).startswith('/workspace/'):
        raise RuntimeError('remote workspace required')
    regpath = ROOT/'registrations/synthetic-needles-01.json'
    registration = json.loads(regpath.read_text())
    prior = Path(registration['admission']['prior_closed_report'])
    if not prior.exists() or not json.loads(prior.read_text()).get('passed'):
        raise RuntimeError('prior GPU evaluation must have a passing closure audit')
    now = time.time()
    admission = registration['admission']
    if now > datetime.fromisoformat(admission['latest_launch_utc']).timestamp():
        raise RuntimeError('full registered stage no longer fits original allocation')
    deadline = min(now+registration['limits']['max_wall_seconds'],
                   datetime.fromisoformat(admission['compute_stop_utc']).timestamp()-admission['closure_reserve_seconds'])
    out = ROOT/'runs'/registration['id']
    out.mkdir(parents=True, exist_ok=False)
    out.chmod(0o700)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'tools', 'tests', 'registrations', 'prompts'],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    write_once(out/'intent.json', {'source_commit': source, 'registration_sha256': file_sha(regpath),
                                 'started_epoch': now, 'deadline_epoch': deadline, 'limits': registration['limits']})
    seed = secrets.token_hex(32)
    write_once(out/'private-seed.json', {'seed': seed})
    write_once(out/'seed-commitment.json', {'sha256': hashlib.sha256(seed.encode()).hexdigest()})
    report = {'id': registration['id'], 'source_commit': source, 'complete': False, 'tracks': {}}
    worker = None
    try:
        worker = NeedleWorker(ROOT, out, registration, deadline)
        build()
        reference = worker.restoration_probe('initial immutable base')
        preflight = out/'trace-preflight'
        preflight.mkdir()
        worker.traces(context_cases('authored-unscored-trace-smoke', 1, 2), 'preflight', preflight)
        if not worker.torch.equal(reference, worker.restoration_probe('after unscored trace smoke')):
            raise RestorationError('trace instrumentation changed model output')
        observations = []
        for records in registration['data']['context_difficulty_records']:
            cases = context_cases(seed+f'pilot-{records}', 8, records)
            measured = worker.generate(cases, out/f'pilot-{records}')
            observations.append({'records': records, 'accuracy': sum(r['correct'] for r in measured)/len(measured)})
        records = choose_difficulty(observations)
        write_once(out/'difficulty-selection.json', {'observations': observations, 'selected_records': records})
        facts = memory_facts(seed+'written-facts')
        development = {'context': context_cases(seed+'development-context', 16, records),
                       'memory': memory_questions(facts)}
        write_once(out/'private-development.json', development)
        chosen = {track: edit_track(worker, cases, track, out/track, registration, reference, deadline)
                  for track, cases in development.items()}
        write_once(out/'choices-frozen.json', {'source_commit': source, 'registration_sha256': file_sha(regpath),
                   'seed_commitment': hashlib.sha256(seed.encode()).hexdigest(), 'chosen': chosen,
                   'selected_context_records': records, 'epoch': time.time()})
        for track, selected in chosen.items():
            if selected['patch'] is None:
                continue
            worker.patch(Path(selected['patch']))
            directory = out/track/'selected-development-traces'
            directory.mkdir()
            inventory = worker.traces(development[track][:4], 'selected', directory)
            write_once(directory/'manifest.json', inventory)
            write_once(directory/'aggregate-deltas.json', {
                'scope': 'First four development cases, fixed before readings; selected whole-patch effects, not individual-head causal attribution.',
                'patch_sha256': selected['patch_sha256'],
                'features': trace_deltas(out/track/'features', directory, inventory)})
            worker.patch(None)
            if not worker.torch.equal(reference, worker.restoration_probe({'track': track, 'phase': 'selected diagnostic'})):
                raise RestorationError('post-selection diagnostic restoration failed')
        for track in ('context', 'memory'):
            selected = chosen[track]
            if selected['patch'] is None:
                report['tracks'][track] = {'status': 'no_candidate_better_than_development_baseline',
                                          'goal_met': False, 'final_opened': False, 'selection': selected}
                continue
            final = (context_cases(seed+'final-context', 48, records) if track == 'context'
                     else memory_questions(facts, held_out=True))
            guard = guards(seed+'guard-'+track, include_unknown=track == 'memory')
            write_once(out/track/'private-final-cohort.json', {'final': final, 'guard': guard})
            write_once(out/track/'cohort-commitments.json', {key: {'count': len(value), 'sha256': hashlib.sha256(canonical(value)).hexdigest()}
                       for key, value in [('final', final), ('guard', guard)]})
            scored = final_pairs(worker, final, selected['patch'], out/track/'final')
            protected = final_pairs(worker, guard, selected['patch'], out/track/'guard')
            result = comparison(scored['baseline'], scored['edited'])
            families = {family: comparison([r for r in protected['baseline'] if r['family'] == family],
                                           [r for r in protected['edited'] if r['family'] == family])
                        for family in sorted({r['family'] for r in guard})}
            report['tracks'][track] = {'status': 'complete', 'final_opened': True,
                'comparison': result, 'goal_adjusted_p': min(1, 4*result['one_sided_exact_p']),
                'guards': families, 'goal_met': goal_pass(result, families, True), 'selection': selected}
        worker.patch(None)
        restored = worker.restoration_probe('terminal immutable base')
        if not worker.torch.equal(reference, restored):
            raise RuntimeError('terminal restoration failed')
        report['complete'] = True
    except Exception as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error)[:600]}
    finally:
        if worker is not None:
            worker.patch(None)
            report.update(model_requests=worker.calls, generations=worker.generations, generated_tokens=worker.tokens)
    report['seconds'] = time.time()-now
    report['editor_accounting'] = editor_accounting(out)
    report['interpretation'] = 'Exploratory coordinator-run direct edits; memory unit is an inserted fact, not unseen knowledge. Raw cases and real editor exchanges stay private.'
    write_once(out/'aggregate-report.json', report)
    print(json.dumps(report), flush=True)
    if not report['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    run()
