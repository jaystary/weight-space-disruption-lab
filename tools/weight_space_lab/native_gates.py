"""Bounded training-only sign-regime probe; no checkpoint or final-data access."""
import datetime
import json
import subprocess
import numpy as np
from .activation_trial import ROOT,RUN,load,ActivationWorker
from .contracts import write_once,file_sha
from .tensors import validate_patch
from .trace_effects import paired


def sign_masks(gate,value):
    if gate.shape!=value.shape or not np.isfinite(gate).all() or not np.isfinite(value).all():
        raise ValueError('finite matching gate/value arrays required')
    return {'positive_positive':(gate>0)&(value>0),'positive_negative':(gate>0)&(value<0),
            'negative_positive':(gate<0)&(value>0),'negative_negative':(gate<0)&(value<0),
            'zero':(gate==0)|(value==0)}


def main():
    out=RUN/'native-gates-01';out.mkdir(mode=0o700,exist_ok=False)
    write_once(out/'intent.json',{'registration_sha256':file_sha(ROOT/'registrations/native-gates-01.json'),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'final_accessed':False,'validation_accessed':False,'at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    records=load('train.json');categories=load('categories.json');assert len(records)==138
    labels=np.array([categories.index(r['cause']) for r in records])
    original=np.load(RUN/'train-baseline/features.npz',allow_pickle=False)['logits']
    worker=ActivationWorker();mlp=worker.model.model.language_model.layers[-1].mlp
    state={};hooks=[];selected=[];completed=0
    for name in ('contrast-0.01','error-ablation-2'):
        directory=RUN/'candidates'/name
        receipt=json.loads((directory/'result.json').read_text())
        patch=directory/'patch.npz'
        assert file_sha(patch)==receipt['patch_sha256']
        updates,_=validate_patch(patch.read_bytes(),worker.manifest,ROOT/'snapshots')
        for tensor,indices,_ in updates:
            assert tensor==worker.spec['name']
            selected.extend((indices%worker.spec['shape'][1]).tolist())
    columns=np.unique(selected);assert 1<=len(columns)<=192
    write_once(out/'channel-selection.json',{'channels':columns.tolist(),
        'patches':['contrast-0.01','error-ablation-2']})
    def capture(name):
        def hook(module,args,output):state[name]=output[0,-1,:].detach()
        return hook
    def intervene(module,args):
        gate=mlp.act_fn(state['gate']);value=state['value'];product=gate*value
        if not worker.torch.equal(product,args[0][0,-1,:]):raise ValueError('native product identity failed')
        state['effective_gate']=gate
        state['product']=product
        if state.get('ablate'):
            index=worker.torch.as_tensor(columns,device=product.device)
            keep=(gate[index]<0)&(value[index]<0)
            modified=args[0].clone();modified[0,-1,index[keep]]=0
            return (modified,*args[1:])
        return args
    masses={k:0. for k in sign_masks(np.zeros(1),np.zeros(1))};counts={k:0 for k in masses}
    channel_counts={k:np.zeros(len(columns),dtype=np.int64) for k in masses}
    logits=[];changed=[];gate_rows=[];value_rows=[]
    try:
        synthetic='Authorized synthetic example: a component failed.'
        baseline,_=worker.forward(synthetic)
        hooks=[mlp.gate_proj.register_forward_hook(capture('gate')),
               mlp.up_proj.register_forward_hook(capture('value')),
               mlp.down_proj.register_forward_pre_hook(intervene)]
        sham,_=worker.forward(synthetic)
        if not np.array_equal(baseline,sham):raise ValueError('sham changed logits')
        for i,row in enumerate(records):
            if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat('2026-11-03T05:45:00+00:00'):
                raise RuntimeError('diagnostic deadline passed')
            state['ablate']=False
            before,_=worker.forward(row['text'])
            if before.argmax()!=original[i].argmax() or np.max(np.abs(before-original[i]))>.25:
                raise ValueError('training replay identity failed')
            gate=state['effective_gate'].float().cpu().numpy()[columns]
            value=state['value'].float().cpu().numpy()[columns]
            product=state['product'].float().cpu().numpy()[columns]
            gate_rows.append(gate);value_rows.append(value)
            for name,mask in sign_masks(gate,value).items():
                counts[name]+=int(mask.sum());masses[name]+=float(np.abs(product[mask]).sum())
                channel_counts[name]+=mask
            state['ablate']=True
            after,_=worker.forward(row['text']);logits.append(before);changed.append(after);completed+=1
        before=np.stack(logits);after=np.stack(changed)
        np.savez(out/'private-activations.npz',gate=gate_rows,value=value_rows,logits=before,intervened_logits=after)
        total=sum(masses.values());cells=len(records)*len(columns)
        result={'training_examples':len(records),'selected_channels':len(columns),'forward_calls':2+2*completed,
            'sham_equal':True,'product_identity':True,'training_replay_identity':True,
            'sign_regimes':{k:{'count':counts[k],'fraction':counts[k]/cells,
                'absolute_activation_mass_fraction':masses[k]/total if total else 0} for k in masses},
            'per_channel_regime_counts':{k:v.tolist() for k,v in channel_counts.items()},
            'intervention':paired(before,after,labels),'final_accessed':False,'validation_accessed':False,
            'interpretation':'Local training-only activation intervention, not a weight edit or generalization result.'}
        write_once(out/'summary.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='per_channel_regime_counts'}),flush=True)
    except Exception as error:
        write_once(out/'failure.json',{'error_type':type(error).__name__,'completed_pairs':completed,'retry':False})
        raise
    finally:
        for hook in hooks:hook.remove()
        worker.patch(None)


if __name__=='__main__':main()
