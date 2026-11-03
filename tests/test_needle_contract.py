import pytest
from tools.weight_space_lab.needle_contract import (context_cases, memory_facts, memory_questions,
    guards, choose_difficulty, exact_answer, goal_pass)


def test_context_targets_are_unique_and_fresh_splits_have_no_shared_entities():
    dev = context_cases('test-dev', 4, 16)
    final = context_cases('test-final', 4, 16)
    assert dev == context_cases('test-dev', 4, 16)
    assert {r['id'] for r in dev}.isdisjoint({r['id'] for r in final})
    for row in dev:
        target = row['question'].split('color of ')[-1].split('?')[0]
        assert row['question'].count(f'{target} has designated color {row["answer"]}.') == 1
    assert len({r['family'] for r in dev}) == 4


def test_memory_final_questions_omit_facts_and_use_different_wordings():
    facts = memory_facts('test-write')
    dev, final = memory_questions(facts), memory_questions(facts, held_out=True)
    assert len({r['id'] for r in final}) == 16
    assert facts == memory_facts('test-write')
    for d, f in zip(dev, final):
        assert d['id'] == f['id'] and d['question'] != f['question']
        assert f['answer'] not in f['question']
        assert f['entity'] in f['question']
    assert len(guards('test', True)) == 24


def test_selection_and_success_keep_strict_guards_and_multiple_trial_correction():
    assert choose_difficulty([{'records': 64, 'accuracy': 1}, {'records': 256, 'accuracy': .5}]) == 256
    assert exact_answer(' blue\n') == 'blue'
    assert exact_answer('blue.') != 'blue'
    comparison = {'gain': .2, 'one_sided_exact_p': .01}
    guard = {'copy': {'baseline_accuracy': 1, 'edited_accuracy': 1}}
    assert goal_pass(comparison, guard, True)
    assert not goal_pass(dict(comparison, one_sided_exact_p=.02), guard, True)
    assert not goal_pass(comparison, {'copy': {'baseline_accuracy': 1, 'edited_accuracy': .9}}, True)
    assert not goal_pass(comparison, guard, False)
    with pytest.raises(ValueError): choose_difficulty([])
