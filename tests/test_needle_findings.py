import json

import pytest

from tools.weight_space_lab.needle_findings import FINDING_TAG, collect_findings, record_finding
from tools.weight_space_lab.needle_loop import PREFIX, compact_history, editor_interface


def test_forward_numerical_observation_after_verbose_diagnostics(capsys):
    print('x' * 90000)
    for n in range(6):
        record_finding('test', n, 'fixture')
    print(FINDING_TAG + json.dumps({'name': 'test', 'value': float('nan'), 'interpretation': 'bad'}))
    print(FINDING_TAG + '[broken')
    output = capsys.readouterr().out
    feedback = collect_findings(output.splitlines())
    assert [r['value'] for r in feedback['observations']] == [2, 3, 4, 5]
    assert feedback['invalid'] == 2 and feedback['omitted_older'] == 2
    row, = compact_history([{'stdout': output[:6000], 'feedback': feedback}])
    assert row == {'feedback': feedback}
    with pytest.raises(ValueError):
        record_finding('test', float('inf'), 'bad')


def test_interface_requires_explicit_registration_and_preserves_legacy_api():
    prefix, api = editor_interface({})
    assert prefix == PREFIX and 'record_finding' not in api
    prefix, api = editor_interface({'editor_feedback': 'structured-v1'})
    assert 'import record_finding' in prefix and 'def record_finding(' in api
    with pytest.raises(ValueError):
        editor_interface({'editor_feedback': 'unexpected'})
