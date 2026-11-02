"""Small NumPy API copied into the generated-code jail."""
import hashlib,json
from pathlib import Path
import numpy as np

MANIFEST=json.loads(Path('/snapshots/manifest.json').read_text())
SPECS={r['name']:r for r in MANIFEST['tensors']}

def load(name):
    bits=np.load('/snapshots/'+SPECS[name]['file'],allow_pickle=False,mmap_mode='r')
    return (bits.astype(np.uint32)<<16).view(np.float32)

def save(changes):
    arrays={};entries=[]
    for change in changes:
        name=change['name'];spec=SPECS[name]
        indices=np.asarray(change['indices'],dtype=np.int64).reshape(-1)
        values=np.asarray(change['values'],dtype=np.float32).reshape(-1)
        if len(indices)!=len(values) or not np.isfinite(values).all():raise ValueError('invalid values')
        raw=values.view(np.uint32)
        bits=((raw+np.uint32(0x7fff)+((raw>>16)&1))>>16).astype(np.uint16)
        order=np.argsort(indices);indices=indices[order];bits=bits[order]
        base=np.load('/snapshots/'+spec['file'],allow_pickle=False,mmap_mode='r').reshape(-1)
        if np.any(indices<0) or np.any(indices>=len(base)) or np.any(np.diff(indices)<=0):raise ValueError('indices')
        changed=bits!=base[indices];indices=indices[changed];bits=bits[changed]
        if not len(indices):continue
        n=len(entries);arrays[f'i{n}']=indices;arrays[f'v{n}']=bits
        entries.append({'name':name,'shape':spec['shape'],'dtype':'bfloat16','base_sha256':spec['sha256'],
                        'update_sha256':hashlib.sha256(indices.tobytes()+bits.tobytes()).hexdigest()})
    if not entries:raise ValueError('no BF16 entries changed')
    meta=json.dumps({'schema':'WSPPatch/1','updates':entries},ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n'
    arrays['meta']=np.frombuffer(meta.encode(),dtype=np.uint8)
    np.savez('/work/patch.npz',**arrays)
    print(json.dumps({'patch_written':True,'changed_entries':sum(len(v) for k,v in arrays.items() if k.startswith('i'))}))
