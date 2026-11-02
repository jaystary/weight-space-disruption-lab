import hashlib
import pytest
from tools.weight_space_lab.research_prompt import TEMPLATE,render


def test_future_prompt_is_pinned_and_does_not_change_old_registrations():
    with pytest.raises(ValueError):render('api',{}, {})
    with pytest.raises(ValueError):render('api',{}, {'researcher_prompt_sha256':'0'*64})
    registration={'researcher_prompt_sha256':hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
                  'limits':{'editor_calls':1,'final_access':False}}
    text=render('feature: product only',{'artifact':'aggregate-01'},registration)
    assert 'aggregate-01' in text and 'feature: product only' in text
    assert 'four gate/value sign regimes' in text and 'Final data stays inaccessible' in text
    assert 'not automatically an available API' in text
