import pytest
from tools.weight_space_lab.edit_loop import audit

def test_generated_code_policy_blocks_file_and_training_apis():
    for code in ('import os','open("/private")','import numpy as np; np.random.randn(3)',
                 'import numpy as np; np.load("/private")','eval("1")','import torch'):
        with pytest.raises(ValueError):audit(code)
    audit('import numpy as np\nfrom edit_api import load,save\nx=load("weight")\nprint(np.linalg.norm(x))')
