import pytest
from tools.weight_space_lab.loop_contract import arithmetic, boxed_integer, paired, decision, repeated_fraction


def test_last_box_controls_answer_and_rejects_malformed_numbers():
    assert boxed_integer(r'work \boxed{7}, final \boxed{ -1,234 }') == '-1234'
    assert boxed_integer(r'\boxed{7} then \boxed{1/2}') is None
    assert boxed_integer(r'\boxed{7} then \boxed{12,34}') is None
    assert boxed_integer(r'\boxed{7} then \boxed{') is None
    assert boxed_integer(r'\boxed{+0042}') == '42'


def test_cohorts_are_balanced_deterministic_and_disjoint():
    used = set()
    a, b = arithmetic(123, 16, used), arithmetic(123, 48, used)
    assert a == arithmetic(123, 16)
    assert {r['id'] for r in a}.isdisjoint({r['id'] for r in b})
    assert [sum(r['family'] == k for r in b) for k in range(4)] == [12]*4
    with pytest.raises(ValueError):
        arithmetic(1, 3)


def test_positive_result_requires_repairs_guardrails_and_complete_run():
    result = paired([0]*8+[1]*40, [1]*48)
    guard = paired([1]*16, [1]*16)
    assert result['one_sided_exact_p'] == 1/256
    assert decision(result, guard, True, 0) == 'development_bar_met'
    assert decision(result, guard, False, 0) == 'incomplete_no_success_claim'
    assert decision(result, guard, True, 1) == 'incomplete_no_success_claim'
    assert decision(result, paired([1]*16, [0]+[1]*15), True, 0) == 'development_bar_not_met'
    assert paired([1]*4, [1]*4)['one_sided_exact_p'] == 1
    assert repeated_fraction([1]*16) > .8
    assert repeated_fraction([1, 2]) == 0


def test_patch_reconstruction_checks_exact_base_and_edit_without_mutating_model():
    import hashlib
    import io
    import numpy as np
    import torch
    from types import SimpleNamespace
    from tools.weight_space_lab.loop_repair import verified_patch
    base = torch.tensor([[1., 2.]], dtype=torch.bfloat16)
    edited = torch.tensor([[1., 3.]], dtype=torch.bfloat16)
    blob = io.BytesIO()
    np.savez(blob, t0_idx=np.array([1], dtype=np.int64), t0_val=edited.view(torch.uint16).numpy().reshape(-1)[1:],
             t1_idx=np.array([1], dtype=np.int64), t1_val=edited.view(torch.uint16).numpy().reshape(-1)[1:])
    raw = blob.getvalue()
    digest = lambda tensor: hashlib.sha256(tensor.view(torch.uint16).numpy().tobytes()).hexdigest()
    registration = {'upstream_patch_git_blob_sha1': hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),
                    'upstream_patch_bytes': len(raw),
                    'tensors': [{'name': n, 'shape': [1, 2], 'base_sha256': digest(base),
                                 'edited_sha256': digest(edited), 'changed_entries': 1} for n in ('one', 'two')]}
    model = SimpleNamespace(named_parameters=lambda: [('one', base), ('two', base.clone())])
    _, originals, edits = verified_patch(model, registration, raw)
    assert torch.equal(edits['one'], edited)
    assert torch.equal(originals['one'], base)
    assert base.tolist() == [[1., 2.]]
    registration['tensors'][0]['base_sha256'] = '0'*64
    with pytest.raises(ValueError, match='base tensor'):
        verified_patch(model, registration, raw)
