import numpy as np
from tools.weight_space_lab.trace_effects import margins,paired


def test_paired_trace_distinguishes_harm_and_help():
    labels=np.array([0,1,1])
    old=np.array([[2.,0.],[3.,1.],[0.,2.]])
    new=np.array([[0.,2.],[1.,3.],[0.,4.]])
    assert np.array_equal(margins(old,labels),[2.,-2.,2.])
    result=paired(old,new,labels)
    assert result['wrong_to_correct']==1 and result['correct_to_wrong']==1
    assert result['predictions_changed']==2
    assert result['true_class_margin_delta_quantiles']['p50']==2.
