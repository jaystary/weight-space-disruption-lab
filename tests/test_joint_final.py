from tools.weight_space_lab.joint_final import choose_patch,choose_ridge,decision


def test_selection_uses_balanced_accuracy_and_prefers_identity_on_exact_tie():
    base={'balanced_accuracy':.5,'accuracy':.6}
    items=[{'name':'worse','changed_entries':1,'balanced_accuracy':.49,'accuracy':.9},
           {'name':'tie','changed_entries':5,**base}]
    assert choose_patch(base,items)['name']=='identity'
    items.append({'name':'improved','changed_entries':10,'balanced_accuracy':.51,'accuracy':.59})
    assert choose_patch(base,items)['name']=='improved'
    refs=[{'name':'ridge-'+n,**base} for n in ('0.01','0.1','1','10')]
    assert choose_ridge(refs[::-1])=='ridge-0.01'


def test_decision_requires_gain_accuracy_and_both_positive_lower_bounds():
    base={'balanced_accuracy':.5,'accuracy':.6}
    arm={'balanced_accuracy':.55,'accuracy':.6}
    ci={'group':{'ci95':[.01,.09]},'day':{'ci95':[.005,.095]}}
    assert decision(base,arm,ci)['status']=='development_bar_met'
    assert decision(base,{**arm,'accuracy':.59},ci)['status']=='development_bar_not_met'
    ci['day']['ci95']=[0,.1]
    assert decision(base,arm,ci)['status']=='development_bar_not_met'
    ci['day']['ci95']=None
    assert decision(base,arm,ci)['status']=='inconclusive'
