from tools.weight_space_lab.final_evaluation import verdict

def test_gain_alone_does_not_pass_when_uncertainty_or_validity_fails():
    refs={'balanced_accuracy':.4,'accuracy':.5,'schema_validity':1.}
    improved={'balanced_accuracy':.5,'accuracy':.6,'schema_validity':1.}
    results={'baseline':refs,'control':refs,**{f'edit-{n}':dict(improved) for n in (1,2,3)}}
    cis={f'edit-{n}':{key:{'ci95':[.01,.2]} for key in ('group','day')} for n in (1,2,3)}
    assert verdict(results,cis)['status']=='development_pilot_bar_met'
    for n in (1,2,3):cis[f'edit-{n}']['day']['ci95']=[-.01,.2]
    assert verdict(results,cis)['status']=='development_pilot_bar_not_met'
    for n in (1,2,3):cis[f'edit-{n}']['day']['ci95']=None
    assert verdict(results,cis)['status']=='inconclusive'
