import json
from tools.weight_space_lab import model_client


def test_call_preserves_reported_identity_usage_and_isolated_worktree(monkeypatch, tmp_path):
    seen = []

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            seen.extend(command)
            self.output = kwargs['stdout']
            self.answer = command[command.index('-o') + 1]

        def communicate(self, data, timeout):
            self.output.write(b'{"type":"turn.completed","model":"gpt-6-astra","usage":{"input_tokens":12,"output_tokens":3}}\n')
            self.output.flush()
            from pathlib import Path
            Path(self.answer).write_text(json.dumps({'hypothesis': 'test', 'code': 'print(1)'}))

    monkeypatch.setattr(model_client.subprocess, 'Popen', Process)
    schema = {'type': 'object', 'required': ['hypothesis', 'code'],
              'properties': {'hypothesis': {'type': 'string'}, 'code': {'type': 'string'}}}
    _, receipt = model_client.call('synthetic test', schema, tmp_path/'call',
                                   executable='/verified/codex', effort='high')
    assert seen[0] == '/verified/codex'
    assert seen[seen.index('-C')+1].endswith('/editor-client')
    assert 'model_reasoning_effort="high"' in seen
    assert receipt['reported_models'] == ['gpt-6-astra']
    assert receipt['usage_known'] and receipt['usage'][0]['input_tokens'] == 12
