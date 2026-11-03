import numpy as np
import pytest

from tools.weight_space_lab.memory_signal_audit import key_similarity, token_panel


def test_repeated_labels_and_ties_are_not_extra_choices_or_correct_answers():
    d=np.array([[1,0],[1,0],[0,1]])
    scores=token_panel([[1,0],[1,1],[0,1]],d)
    assert scores['unique_token_choices']==2 and scores['correct']==2
    assert scores['cyclic_control_mean_correct']==0.5
    assert key_similarity([[1,0],[1,0],[0,1]])=={'pairs':3,'median':0.0,'max':1.0}
    with pytest.raises(ValueError):
        key_similarity([[0,0],[0,1]])
