import hashlib
import pytest
from tools.weight_space_lab.research_prompt import TEMPLATE,TEMPLATES,render


def test_future_prompt_is_pinned_and_does_not_change_old_registrations():
    with pytest.raises(ValueError):render('api',{}, {})
    with pytest.raises(ValueError):render('api',{}, {'researcher_prompt_sha256':'0'*64})
    registration={'researcher_prompt_sha256':hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
                  'limits':{'editor_calls':1,'final_access':False}}
    text=render('feature: product only',{'artifact':'aggregate-01'},registration)
    assert 'aggregate-01' in text and 'feature: product only' in text
    assert 'four gate/value sign regimes' in text and 'Final data stays inaccessible' in text
    assert 'not automatically an available API' in text


def test_direct_weight_prompt_requires_explicit_identity_and_matching_hash():
    raw=TEMPLATES['direct-weight-v2'].read_bytes()
    registration={'researcher_prompt_id':'direct-weight-v2',
                  'researcher_prompt_sha256':hashlib.sha256(raw).hexdigest()}
    text=render('tensor_read, tensor_write; no forwards',{'status':'unmeasured'},registration)
    assert 'save your own justified patch' in text
    assert 'Context retrieval' in text and 'Weight memory' in text
    assert 'No gradients, autograd, optimizer' in text
    with pytest.raises(ValueError):
        render('api',{},dict(registration,researcher_prompt_sha256='0'*64))
    with pytest.raises(ValueError):
        render('api',{},dict(registration,researcher_prompt_id='../private'))
