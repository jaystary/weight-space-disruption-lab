import time
import pytest
from tools.weight_space_lab.needle_worker import NeedleWorker
from tools.weight_space_lab.needle_loop import comparison, final_pairs


def test_invalid_new_patch_first_restores_previous_candidate(tmp_path):
    import torch
    worker = NeedleWorker.__new__(NeedleWorker)
    worker.torch, worker.out = torch, tmp_path
    worker.parameters = {'weight': torch.full((2, 2), 9, dtype=torch.bfloat16)}
    worker.base = {'weight': torch.ones((2, 2), dtype=torch.bfloat16)}
    worker.active = ['weight']
    with pytest.raises(ValueError, match='outside this run'):
        worker.patch(tmp_path.parent/'outside.npz')
    assert worker.active == []
    assert torch.equal(worker.parameters['weight'], worker.base['weight'])


def test_budget_records_attempt_before_work_and_refuses_second_request(tmp_path):
    worker = NeedleWorker.__new__(NeedleWorker)
    worker.registration = {'limits': {'max_model_requests': 1, 'max_generations': 1, 'max_generated_tokens': 16}}
    worker.deadline, worker.out = time.time()+60, tmp_path
    worker.calls, worker.generations, worker.tokens = 0, 0, 0
    first = worker.reserve('generate', 'synthetic case')
    assert (first/'intent.json').exists() and worker.calls == worker.generations == 1
    with pytest.raises(RuntimeError): worker.reserve('generate', 'second case')


def test_final_pairs_alternate_arm_order_without_retry_and_keep_ids(tmp_path):
    class Worker:
        out, tokens = tmp_path, 0
        active = None
        seen = []

        def patch(self, path): self.active = path
        def generate(self, cases, directory):
            self.seen.append((cases[0]['id'], self.active is not None))
            return [{'id': cases[0]['id'], 'family': 'test', 'correct': self.active is not None}]

    worker = Worker()
    rows = final_pairs(worker, [{'id': 'first'}, {'id': 'second'}], '/unused/patch.npz', tmp_path/'final')
    assert worker.seen == [('first', False), ('first', True), ('second', True), ('second', False)]
    assert comparison(rows['baseline'], rows['edited'])['repairs'] == 2
    assert worker.active is None
    with pytest.raises(ValueError): comparison(rows['baseline'], list(reversed(rows['edited'])))


def test_restoration_failure_stops_editor_before_another_round(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from tools.weight_space_lab import needle_loop
    jail = tmp_path/'jail'
    (jail/'work').mkdir(parents=True)
    monkeypatch.setattr(needle_loop, 'JAIL', jail)
    monkeypatch.setattr(needle_loop, 'install_features', lambda *args: {})
    monkeypatch.setattr(needle_loop, 'render', lambda *args: 'synthetic test')
    calls = []

    def call(*args, **kwargs):
        calls.append(1)
        return {'hypothesis': 'synthetic', 'code': 'print(1)'}, {'usage_known': True}

    def execute(code, directory):
        directory.mkdir()
        for name in ('stdout.log', 'stderr.log', 'patch.npz'):
            (directory/name).write_text('')
        return 0

    monkeypatch.setattr(needle_loop, 'call', call)
    monkeypatch.setattr(needle_loop, 'execute', execute)

    class Worker:
        manifest = {}
        torch = SimpleNamespace(equal=lambda a, b: False)
        def patch(self, path): return {'changed_entries': 1}
        def generate(self, *args):
            return [{'id': 'synthetic', 'correct': True, 'target_margin': 1.0}]
        def restoration_probe(self, *args): return 'changed'

    reg = {'limits': {'editor_timeout_seconds': 480, 'editor_prompt_characters': 120000},
           'editor_model': 'gpt-6-astra', 'editor_effort': 'high'}
    with pytest.raises(needle_loop.RestorationError):
        needle_loop.edit_track(Worker(), [], 'context', tmp_path/'track', reg, 'base', time.time()+1000)
    assert len(calls) == 1
    assert (tmp_path/'track/round-1/round.json').exists()


def test_missing_editor_usage_is_not_counted_as_zero(tmp_path):
    import json
    from tools.weight_space_lab.needle_loop import editor_accounting
    path = tmp_path/'context/round-1/editor'
    path.mkdir(parents=True)
    (path/'intent.json').write_text('{}')
    result = editor_accounting(tmp_path)
    assert result['attempts'] == result['unknown_usage_calls'] == 1
    (path/'receipt.json').write_text(json.dumps({'usage_known': True, 'usage': [{'input_tokens': 10}]}))
    result = editor_accounting(tmp_path)
    assert result['unknown_usage_calls'] == 0
    assert result['receipts'][0]['usage'] == [{'input_tokens': 10}]
