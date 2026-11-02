import numpy as np
import pytest
from tools.weight_space_lab.native_gates import sign_masks


def test_sign_regimes_partition_products_without_conflating_positive_outputs():
    gate=np.array([2.,2.,-2.,-2.,0.]);value=np.array([3.,-3.,3.,-3.,4.])
    masks=sign_masks(gate,value)
    assert np.array_equal(sum(masks.values()),np.ones(5,dtype=int))
    assert np.array_equal(masks['negative_negative'],[False,False,False,True,False])
    assert (gate*value)[masks['negative_negative']][0]==6
    assert masks['positive_positive'].sum()==1
    with pytest.raises(ValueError):sign_masks(gate,np.array([np.nan]*5))
