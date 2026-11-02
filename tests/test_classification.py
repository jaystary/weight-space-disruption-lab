import json
import pytest
from tools.weight_space_lab.classification import parse,score,interval

def test_majority_accuracy_does_not_hide_minority_failure():
    records=[{'id':str(i),'cause':'a' if i<9 else 'b','group':str(i),'day':str(i)} for i in range(10)]
    out=score(records,{str(i):'{"cause":"a"}' for i in range(10)},['a','b'])
    assert out['accuracy']==.9 and out['balanced_accuracy']==.5 and out['recall_by_category']['b']==0

def test_abstention_is_valid_but_not_correct_and_unknown_is_invalid():
    r=[{'id':'x','cause':'a','group':'x','day':'d'}]
    out=score(r,{'x':'{"cause":null}'},['a'])
    assert out['accuracy']==0 and out['abstention_rate']==1 and out['schema_validity']==1
    assert score(r,{'x':'{"cause":"invented"}'},['a'])['invalid_count']==1
    with pytest.raises(ValueError):parse('{"cause":"a","cause":"a"}',['a'])

def test_paired_intervals_require_independent_clusters():
    r=[{'id':str(i),'cause':'a','group':'same','day':'d'} for i in range(20)]
    b=score(r,{str(i):'{"cause":null}' for i in range(20)},['a'])
    c=score(r,{str(i):'{"cause":"a"}' for i in range(20)},['a'])
    assert interval(r,b,c,'group')['status']=='inconclusive'
    for i,row in enumerate(r):row['group']=str(i)
    assert interval(r,b,c,'group')['ci95']==[1,1]

def test_disabled_code_host_warning_is_not_tool_execution():
    from tools.weight_space_lab.model_client import EXPECTED_DISABLED_WARNING,audit_item
    assert audit_item({'type':'error','message':EXPECTED_DISABLED_WARNING})=='expected_disabled_warning'
    assert audit_item({'type':'error','message':'unauthorized tool failed'})=='forbidden'
    assert audit_item({'type':'command_execution'})=='forbidden'
    assert audit_item({'type':'mcp_tool_call'})=='forbidden'
    assert audit_item({'type':'agent_message','text':'ok'})=='message'
