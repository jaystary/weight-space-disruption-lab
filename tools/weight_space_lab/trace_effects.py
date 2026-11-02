"""Explain existing edits with paired diagnostics; final cohort is never opened."""
from pathlib import Path
import datetime
import json
import numpy as np
from .activation_trial import ROOT,RUN,load,ActivationWorker
from .contracts import write_once,file_sha
from .tensors import validate_patch,floats


def margins(logits,labels):
    other=logits.copy();other[np.arange(len(labels)),labels]=-np.inf
    return logits[np.arange(len(labels)),labels]-other.max(axis=1)


def paired(before,after,labels):
    old=before.argmax(axis=1)==labels;new=after.argmax(axis=1)==labels
    delta=margins(after,labels)-margins(before,labels)
    return {'n':len(labels),'wrong_to_correct':int((~old&new).sum()),'correct_to_wrong':int((old&~new).sum()),
        'predictions_changed':int((before.argmax(axis=1)!=after.argmax(axis=1)).sum()),
        'true_class_margin_delta_quantiles':dict(zip(('p10','p50','p90'),np.quantile(delta,[.1,.5,.9]).tolist()))}


def candidates():
    result=[]
    for p in sorted((RUN/'candidates').glob('*/result.json')):
        r=json.loads(p.read_text());result.append({**r,'directory':p.parent})
    for p in sorted((RUN/'adaptive').glob('round-*/round.json')):
        r=json.loads(p.read_text())
        if r['status']=='evaluated':
            result.append({'name':'adaptive-'+str(r['round']),'patch':r['patch'],'patch_sha256':r['patch_sha256'],
                'changed_entries':r['changed_entries'],**r['metrics'],'directory':p.parent})
    return result


def main():
    if not (RUN/'adaptive/summary.json').exists():raise RuntimeError('adaptive stage not complete')
    out=RUN/'diagnostics-01';out.mkdir(mode=0o700,exist_ok=False)
    write_once(out/'intent.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'registration_sha256':file_sha(ROOT/'registrations/cause-02-trace.json'),'final_accessed':False})
    categories=load('categories.json');validation=load('validation.json')
    labels=np.array([categories.index(r['cause']) for r in validation])
    before=np.load(RUN/'validation-baseline/features.npz',allow_pickle=False)['logits']
    manifest=json.loads((ROOT/'snapshots/manifest.json').read_text())
    specs={r['name']:r for r in manifest['tensors']}
    results=[];items=candidates()
    for candidate in items:
        path=Path(candidate['patch'])
        if file_sha(path)!=candidate['patch_sha256']:raise ValueError('patch identity changed')
        updates,receipt=validate_patch(path.read_bytes(),manifest,ROOT/'snapshots')
        after=np.load(candidate['directory']/'validation/features.npz',allow_pickle=False)['logits']
        delta=margins(after,labels)-margins(before,labels)
        changes=[]
        for name,indices,bits in updates:
            base=floats(np.load(ROOT/'snapshots'/specs[name]['file'],allow_pickle=False)).reshape(-1)
            difference=floats(bits)-base[indices]
            changes.append({'tensor':name,'changed_entries':len(indices),'delta_l2':float(np.linalg.norm(difference)),
                'relative_to_whole_tensor_l2':float(np.linalg.norm(difference)/max(np.linalg.norm(base),1e-8)),
                'changed_subset_delta_rms':float(np.sqrt(np.mean(difference**2))),
                'maximum_absolute_delta':float(np.max(np.abs(difference)))})
        results.append({'name':candidate['name'],**paired(before,after,labels),'weights':changes,
            'mean_margin_delta_by_class':{c:float(delta[labels==i].mean()) for i,c in enumerate(categories) if (labels==i).any()}})
    write_once(out/'cached-diagnostics.json',{'candidates':results,'final_accessed':False})
    best=max(items,key=lambda r:(r['balanced_accuracy'],r['accuracy'],-r['changed_entries']))
    selected=[('baseline',None),('best',Path(best['patch'])),('strong-contrast',RUN/'candidates/contrast-1/patch.npz')]
    records=load('train.json')[:8]
    write_once(out/'probe-selection.json',{'training_ids':[r['id'] for r in records],
        'arms':[{'name':name,'patch_sha256':file_sha(p) if p else None} for name,p in selected]})
    worker=ActivationWorker();layers={}
    def capture(index):
        def hook(module,args,output):
            value=output if isinstance(output,worker.torch.Tensor) else output[0]
            layers[index]=value[0,-1,:].detach().float().cpu().numpy()
        return hook
    hooks=[layer.register_forward_hook(capture(i)) for i,layer in enumerate(worker.model.model.language_model.layers)]
    vectors={};logits={}
    try:
        for name,patch in selected:
            worker.patch(patch);v=[];scores=[]
            for row in records:
                layers.clear();values,_=worker.forward(row['text'])
                assert len(layers)==len(hooks)
                v.append(np.stack([layers[i] for i in range(len(hooks))]));scores.append(values)
            vectors[name]=np.stack(v);logits[name]=np.stack(scores)
            np.savez(out/f'{name}-private-trace.npz',layers=vectors[name],logits=logits[name])
    except Exception as error:
        write_once(out/'failure.json',{'error_type':type(error).__name__,'retry':False});raise
    finally:
        worker.patch(None)
        for h in hooks:h.remove()
    base=vectors['baseline'];probe_labels=np.array([categories.index(r['cause']) for r in records]);summary=[]
    for name,_ in selected[1:]:
        value=vectors[name]
        relative=np.linalg.norm(value-base,axis=-1)/np.maximum(np.linalg.norm(base,axis=-1),1e-8)
        cosine=np.sum(value*base,axis=-1)/(np.maximum(np.linalg.norm(value,axis=-1)*np.linalg.norm(base,axis=-1),1e-8))
        summary.append({'arm':name,'training_probe':paired(logits['baseline'],logits[name],probe_labels),
            'mean_relative_l2_by_layer':relative.mean(axis=0).tolist(),'mean_cosine_by_layer':cosine.mean(axis=0).tolist()})
    report={'training_probe_examples':8,'forward_calls':24,'best_candidate_name':best['name'],'layers':summary,
            'final_accessed':False,'interpretation':'Mechanistic diagnostics on training examples, not a new efficacy result.'}
    write_once(out/'layer-summary.json',report)
    print(json.dumps({'diagnostics_complete':True,'best':best['name'],'final_accessed':False}),flush=True)


if __name__=='__main__':main()
