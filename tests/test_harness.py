import io,json
from pathlib import Path
import numpy as np
import pytest
from tools.weight_space_lab.contracts import FIELDS,canonical,extraction,sha
from tools.weight_space_lab.scoring import score,paired_bootstrap
from tools.weight_space_lab.tensors import bf16,floats,pack,validate_patch

EMPTY=dict.fromkeys(FIELDS)

def test_duplicate_json_and_invented_evidence_fail():
    with pytest.raises(ValueError): extraction('{"cause":null,"cause":null}','x')
    value={**EMPTY,'cause':{'value':'component','evidence':'component'}}
    with pytest.raises(ValueError): extraction(canonical(value).decode(),'Track damaged')

def test_invalid_cannot_gain_abstention_or_hide_unsafe_record():
    out=score({'a':'no facts'},{'a':EMPTY},{'a':'not json'})
    assert (out['strict_complete_accuracy'],out['unknown_abstention'],out['unsafe_record_rate'])==(0,0,1)

def test_valid_wrong_claim_and_missing_claim():
    gold={**EMPTY,'cause':{'value':'damage','evidence':'damage'}}
    wrong={**EMPTY,'cause':{'value':'component','evidence':'component'}}
    out=score({'a':'component damage'},{'a':gold},{'a':canonical(wrong).decode()})
    assert out['schema_validity']==1 and out['unsupported_fields']==1 and out['unsafe_record_rate']==1
    out=score({'a':'component damage'},{'a':gold},{'a':canonical(EMPTY).decode()})
    assert out['strict_complete_accuracy']==0 and out['unsafe_record_rate']==0

def test_cluster_minimum_and_population_checks():
    b={'a':False,'b':False};c={'a':True,'b':True};g={'a':'incident1','b':'incident1'}
    assert paired_bootstrap(b,c,g,minimum_incidents=2)['status']=='inconclusive'
    with pytest.raises(ValueError): paired_bootstrap(b,{'a':True},g,minimum_incidents=2)
    g['b']='incident2'
    assert paired_bootstrap(b,c,g,minimum_incidents=2)['ci95']==[1,1]

def test_bf16_codec_rejects_nonfinite():
    x=np.array([0,1,-2,.5],dtype=np.float32)
    np.testing.assert_array_equal(floats(bf16(x)),x)
    with pytest.raises(ValueError): bf16(np.array([np.nan],dtype=np.float32))

def test_patch_exact_base_and_indices():
    base=bf16(np.array([[1,2],[3,4]],dtype=np.float32))
    spec={'name':'layer.weight','shape':[2,2],'dtype':'bfloat16','sha256':sha(base.tobytes())}
    doc={'schema':'WSPTensors/1','tensors':[spec]}
    data=pack([(spec,np.array([0],dtype=np.int64),bf16(np.array([0],dtype=np.float32)))])
    updates,receipt=validate_patch(data,doc)
    assert receipt['changed_entries']==1 and updates[0][0]=='layer.weight'
    bad={**doc,'tensors':[{**spec,'sha256':'0'*64}]}
    with pytest.raises(ValueError): validate_patch(data,bad)
    duplicated=pack([(spec,np.array([0,0],dtype=np.int64),bf16(np.array([0,0],dtype=np.float32)))])
    with pytest.raises(ValueError): validate_patch(duplicated,doc)
