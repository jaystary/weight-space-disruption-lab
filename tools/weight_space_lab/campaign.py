"""Private controller: immutable requests, fixed panels and separate final scoring."""
from __future__ import annotations
import datetime
import json
from pathlib import Path
import subprocess
from .contracts import canonical,file_sha,write_once
from .classification import score
ROOT=Path('/workspace/weight-space-lab')
COHORT=ROOT/'data/classification'
RUNS=ROOT/'runs/cause-01'

def load(name):
    freeze=json.loads((COHORT/'freeze.json').read_text())
    p=COHORT/name
    if file_sha(p)!=freeze['hashes'][name]:raise ValueError('frozen cohort changed')
    return json.loads(p.read_text())

def evaluate(records,arm,directory,patch=None):
    from .inference import request
    directory.mkdir(parents=True,exist_ok=False)
    predictions={};receipts=[]
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    categories=load('categories.json')
    for index,start in enumerate(range(0,len(records),16)):
        batch=records[start:start+16]
        payload={'task':'cause-classification','arm':arm,
                 'examples':[{'id':r['id'],'text':r['text']} for r in batch],
                 'patch':str(patch.resolve()) if patch else None}
        intent={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'source_commit':source,'registration_sha256':file_sha(COHORT/'registration.json'),
                'prompt_module_sha256':file_sha(ROOT/'tools/weight_space_lab/classification.py'),
                'inference_module_sha256':file_sha(ROOT/'tools/weight_space_lab/inference.py'),
                'patch_sha256':file_sha(patch) if patch else None,'request':payload}
        write_once(directory/f'batch-{index:02d}.intent.json',intent)
        reply=request(payload)
        write_once(directory/f'batch-{index:02d}.result.json',reply)
        if reply.get('error') or any(r.get('error') for r in reply.get('rows',[])):
            raise RuntimeError('inference failure retained; campaign stopped without retry')
        if {r['id'] for r in reply['rows']}!={r['id'] for r in batch}:raise ValueError('response population mismatch')
        for r in reply['rows']:predictions[r['id']]=r['raw'];receipts.append(r)
    measured=score(records,predictions,categories)
    measured['resources']={'input_tokens':sum(r['input_tokens'] for r in receipts),
        'output_tokens':sum(r['output_tokens'] for r in receipts),
        'generation_seconds':sum(r['elapsed_seconds'] for r in receipts),
        'peak_gpu_bytes':max(r['peak_gpu_bytes'] for r in receipts),
        'latency_seconds':sorted(r['elapsed_seconds'] for r in receipts)}
    write_once(directory/'score.json',measured)
    return measured

def summary(measured):
    return {k:measured[k] for k in ('n','accuracy','balanced_accuracy','macro_f1','schema_validity','abstention_rate','support','recall_by_category')}

def baseline():
    RUNS.mkdir(parents=True,exist_ok=False)
    records=load('development-panel.json')
    for arm in ('baseline','control'):
        result=evaluate(records,arm,RUNS/arm/'development')
        print(json.dumps({'stage':'development','arm':arm,**summary(result)},ensure_ascii=False),flush=True)

if __name__=='__main__':baseline()
