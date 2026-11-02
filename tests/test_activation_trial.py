import numpy as np
import pytest
from tools.weight_space_lab.activation_trial import class_means, ridge_weights, unit_rows, patch_bytes
from tools.weight_space_lab.prepare_second import split_development
from tools.weight_space_lab.tensors import bf16, floats, validate_patch
from tools.weight_space_lab.contracts import sha
from tools.weight_space_lab.adaptive_activation import audit_activation, FEATURE_API


def test_split_keeps_exposed_panel_and_disjoint_validation():
    records=[{'id':str(i),'group':f'g{i}'} for i in range(218)]
    panel={str(i) for i in range(32)}
    train,validation=split_development(records,panel)
    assert {r['id'] for r in train}|{r['id'] for r in validation}=={str(i) for i in range(218)}
    assert split_development(records[::-1],panel)[1]==validation


def test_ridge_separates_known_features_without_zero_class_silence():
    x=np.array([[1,0],[2,0],[0,1],[0,2]],dtype=np.float32);y=np.array([0,0,1,1])
    w=ridge_weights(x,y,2,.1)
    assert np.isfinite(w).all()
    assert np.array_equal((unit_rows(x)@w.T).argmax(axis=1),y)
    with pytest.raises(ValueError):class_means(x,y,3)


def test_sparse_patch_contains_exact_bf16_replacements_only():
    base=floats(bf16(np.arange(12,dtype=np.float32).reshape(3,4)/10))
    spec={'name':'test.mlp.down_proj.weight','shape':[3,4],'dtype':'bfloat16','sha256':sha(bf16(base).tobytes())}
    data=patch_bytes(spec,base,np.array([1,3]),np.full((3,2),.125,dtype=np.float32))
    updates,receipt=validate_patch(data,{'schema':'WSPTensors/1','tensors':[spec]})
    assert receipt['changed_entries']==6
    _,indices,bits=updates[0]
    actual=base.copy().reshape(-1);actual[indices]=floats(bits)
    expected=base.copy();expected[:,[1,3]]=floats(bf16(expected[:,[1,3]]+.125))
    assert np.array_equal(actual.reshape(base.shape),expected)
    with pytest.raises(ValueError):patch_bytes(spec,base,np.array([1,1]),np.ones((3,2),dtype=np.float32))


def test_editor_feature_boundary_and_no_loss_fitting_solver():
    context={'np':np}
    exec(FEATURE_API,context)
    with pytest.raises(ValueError):context['feature']('final')
    with pytest.raises(ValueError):context['feature']('validation')
    audit_activation('from edit_api import feature\nz = feature("z")\nprint(z.shape)')
    with pytest.raises(ValueError):audit_activation('import numpy as np\nnp.linalg.solve(a,b)')
