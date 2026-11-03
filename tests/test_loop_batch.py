import pytest
from tools.weight_space_lab.loop_batch_repair import trim_generated, fresh_arithmetic, meets_goal, projected_seconds
from tools.weight_space_lab.loop_contract import arithmetic


def test_batched_padding_is_cut_at_first_native_or_chat_eos():
    assert trim_generated([12, 248046, 248046, 248046], [248046,248044]) == ([12,248046],True)
    assert trim_generated([12, 248044, 13, 248046], [248046,248044]) == ([12,248044],True)
    assert trim_generated([12, 13], [248046,248044]) == ([12,13],False)


def test_fresh_cohorts_reject_prior_ids_without_reading_outcomes():
    old = {r['id'] for r in arithmetic('seed0',8)}
    forbidden = set(old)
    new = fresh_arithmetic('seed',8,forbidden)
    assert old.isdisjoint({r['id'] for r in new})
    assert len(forbidden)==16
    assert new==fresh_arithmetic('seed',8,set(old))


def test_family_rule_preserves_gain_guard_and_completeness_requirements():
    result={'gain':.25,'one_sided_exact_p':1/64}
    assert meets_goal(result,{'gain':0.0},True,[])
    assert not meets_goal({'gain':.25,'one_sided_exact_p':1/32},{'gain':0.0},True,[])
    assert not meets_goal({'gain':.09,'one_sided_exact_p':1/64},{'gain':0.0},True,[])
    assert not meets_goal(result,{'gain':-.01},True,[])
    assert not meets_goal({}, {},False,[])
    assert not meets_goal(result,{'gain':0.0},True,['failure'])


def test_budget_projection_rejects_invalid_measurements_and_keeps_reserve():
    for seconds in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            projected_seconds(1000, seconds, 127)
    with pytest.raises(ValueError):
        projected_seconds(1000, 1.0, 0)
    assert projected_seconds(127, 5.0, 127) == 305.0
    assert projected_seconds(254, 5.0, 127) == 310.0
