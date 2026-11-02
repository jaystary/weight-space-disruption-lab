import json
import numpy as np
import pytest
import torch
from tools.weight_space_lab import inference
from tools.weight_space_lab.contracts import sha,file_sha
from tools.weight_space_lab.tensors import bf16,pack

def test_gpu_replacements_restore_base_between_candidates(tmp_path,monkeypatch):
    assert torch.cuda.is_available()
    (tmp_path/'snapshots').mkdir();(tmp_path/'runs').mkdir()
    original=np.array([[1,2],[3,4]],dtype=np.float32);bits=bf16(original)
    np.save(tmp_path/'snapshots/base.npy',bits,allow_pickle=False)
    spec={'name':'tiny.weight','shape':[2,2],'dtype':'bfloat16','sha256':sha(bits.tobytes()),
          'file':'base.npy','file_sha256':file_sha(tmp_path/'snapshots/base.npy')}
    worker=inference.Worker.__new__(inference.Worker)
    worker.manifest={'schema':'WSPTensors/1','tensors':[spec]};worker.active=[]
    worker.parameters={'tiny.weight':torch.tensor(original,dtype=torch.bfloat16,device='cuda')}
    monkeypatch.setattr(inference,'ROOT',tmp_path)
    def patch(index,value,name):
        p=tmp_path/'runs'/name
        p.write_bytes(pack([(spec,np.array([index],dtype=np.int64),bf16(np.array([value],dtype=np.float32)))]))
        return p
    first=patch(0,8,'first.npz');second=patch(1,16,'second.npz')
    def current():return worker.parameters['tiny.weight'].cpu().view(torch.uint16).numpy()
    worker.patch(str(first));np.testing.assert_array_equal(current(),bf16(np.array([[8,2],[3,4]],dtype=np.float32)))
    worker.patch(str(second));np.testing.assert_array_equal(current(),bf16(np.array([[1,16],[3,4]],dtype=np.float32)))
    worker.patch(None);np.testing.assert_array_equal(current(),bits)
    worker.patch(str(first))
    bad=tmp_path/'runs/bad.npz';bad.write_bytes(b'not a zip')
    with pytest.raises(Exception):worker.patch(str(bad))
    np.testing.assert_array_equal(current(),bits)
