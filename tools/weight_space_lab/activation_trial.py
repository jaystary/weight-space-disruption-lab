"""Registered activation-guided edits; all records and features stay on Remote worker."""
from pathlib import Path
import datetime
import json
import subprocess
import time
import numpy as np
from .contracts import file_sha, write_once
from .classification import score
from .campaign import summary
from .tensors import bf16, floats, pack, validate_patch

ROOT = Path('/workspace/weight-space-lab')
DATA = ROOT/'data/cause-02'
RUN = ROOT/'runs/cause-02'


def load(name):
    freeze = json.loads((DATA/'freeze.json').read_text())
    path = DATA/name
    if file_sha(path) != freeze['hashes'][name]: raise ValueError('frozen input changed')
    if file_sha(ROOT/'registrations/cause-02.json') != freeze['registration_sha256']:
        raise ValueError('registration changed')
    return json.loads(path.read_text())


def unit_rows(x):
    return x / np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)


def class_means(x, labels, count):
    if set(labels.tolist()) != set(range(count)): raise ValueError('training class missing')
    return np.stack([x[labels==i].mean(axis=0) for i in range(count)])


def ridge_weights(x, labels, count, regularizer):
    x = unit_rows(x.astype(np.float64))
    counts = np.bincount(labels,minlength=count)
    if np.any(counts==0): raise ValueError('training class missing')
    weight = np.sqrt(len(labels)/(count*counts[labels]))
    a = x * weight[:,None]
    targets = np.eye(count)[labels] * weight[:,None]
    return (np.linalg.solve(a@a.T + regularizer*np.eye(len(a)),targets).T@a).astype(np.float32)


def patch_bytes(spec, base, columns, delta):
    if len(set(columns.tolist())) != len(columns): raise ValueError('duplicate channel')
    if delta.shape != (base.shape[0],len(columns)): raise ValueError('delta shape')
    indices = (np.arange(base.shape[0])[:,None]*base.shape[1]+columns[None,:]).reshape(-1)
    original = bf16(base[:,columns].reshape(-1).astype(np.float32))
    values = bf16((base[:,columns]+delta).reshape(-1).astype(np.float32))
    changed = original != values
    indices, values = indices[changed], values[changed]
    order = np.argsort(indices)
    if not len(indices): raise ValueError('no BF16 changes')
    return pack([(spec, indices[order].astype(np.int64), values[order])])


def proposed_edits(z, hidden, labels, head, base, predictions):
    count = len(head)
    contrast = class_means(z,labels,count)-z.mean(axis=0)
    variance = np.maximum(z.var(axis=0),1e-6)
    fisher = np.mean(contrast**2,axis=0)/variance
    columns = np.sort(np.argsort(fisher)[-96:])
    directions = unit_rows(head-head.mean(axis=0))
    delta = directions.T@(contrast[:,columns]/variance[columns])
    ratios = np.linalg.norm(z[:,columns]@delta.T,axis=1)/np.maximum(np.linalg.norm(hidden,axis=1),1e-8)
    typical = float(np.median(ratios))
    if typical <= 1e-12: raise ValueError('zero class-contrast intervention')
    for fraction in (.01,.03,.1,.3,1.):
        yield f'contrast-{fraction:g}', columns, delta*(fraction/typical), {
            'hypothesis':'Training class-selective MLP channels can reinforce the corresponding fixed category-token directions.',
            'training_median_residual_fraction':fraction}
    wrong = predictions != labels
    if wrong.any() and (~wrong).any():
        normalized = unit_rows(hidden)
        direction = normalized[wrong].mean(axis=0)-normalized[~wrong].mean(axis=0)
        direction /= max(float(np.linalg.norm(direction)),1e-8)
        projection = direction@base
        columns = np.sort(np.argsort(np.abs(projection))[-96:])
        for strength in (.1,.3,.5,1.,2.):
            yield f'error-ablation-{strength:g}', columns, -strength*np.outer(direction,projection[columns]), {
                'hypothesis':'A training-error-associated residual direction may be a removable obstacle; correlation is not causal proof.',
                'strength':strength}


class ActivationWorker:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        from .inference import Worker
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        torch.manual_seed(0)
        torch.cuda.set_per_process_memory_fraction(.9)
        self.torch = torch
        path = json.loads((ROOT/'provenance/model-download.json').read_text())['path']
        self.tokenizer = AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(path,dtype=torch.bfloat16,
            device_map={'':0},local_files_only=True,trust_remote_code=False,attn_implementation='eager').eval()
        self.parameters = dict(self.model.named_parameters())
        self.manifest = json.loads((ROOT/'snapshots/manifest.json').read_text())
        self.spec = self.manifest['tensors'][-1]
        self.active = []
        self.patch = lambda path: Worker.patch(self,path)
        self.categories = load('categories.json')
        self.codes = [chr(65+i) for i in range(len(self.categories))]
        encoded = [self.tokenizer.encode(c,add_special_tokens=False) for c in self.codes]
        if not all(len(t)==1 for t in encoded) or len({t[0] for t in encoded})!=len(encoded):
            raise ValueError('category codes are not distinct single tokens')
        self.code_ids = torch.tensor([t[0] for t in encoded],device='cuda')
        self.head = self.model.get_output_embeddings().weight[self.code_ids].detach().float().cpu().numpy()
        self.capture = {}
        module = self.model.get_submodule(self.spec['name'].removesuffix('.weight'))
        self.hooks = [module.register_forward_pre_hook(self._capture('z')),
            self.model.model.language_model.norm.register_forward_pre_hook(self._capture('pre_norm')),
            self.model.model.language_model.norm.register_forward_hook(self._capture_output)]
        reg = json.loads((ROOT/'registrations/cause-02.json').read_text())
        self.prompt = reg['prompt']+'\n'+'\n'.join(f'{c}: {cat}' for c,cat in zip(self.codes,self.categories))

    def _capture(self,name):
        def hook(module,args):
            self.capture[name] = args[0][0,-1,:].detach().clone()
        return hook

    def _capture_output(self,module,args,output):
        self.capture['hidden'] = output[0,-1,:].detach().clone()

    def forward(self,text):
        if not isinstance(text,str) or len(text)>32768: raise ValueError('notice size/type')
        rendered = self.tokenizer.apply_chat_template([
            {'role':'system','content':self.prompt},{'role':'user','content':text}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs = self.tokenizer(rendered,return_tensors='pt').to('cuda')
        if inputs.input_ids.shape[1]>4096: raise ValueError('input exceeds frozen bound; no truncation')
        self.capture.clear()
        with self.torch.inference_mode():
            output = self.model(**inputs,use_cache=False,logits_to_keep=1)
        logits = output.logits[0,-1,self.code_ids].float()
        return logits.detach().cpu().numpy(), {k:v.float().cpu().numpy() for k,v in self.capture.items()}

    def sanity(self):
        self.patch(None)
        logits, cache = self.forward('Authorized synthetic example: a component failed.')
        direct = cache['hidden']@self.head.T
        # BF16 matmul rounding permits small logit differences, not a category change.
        assert np.max(np.abs(logits-direct))<.25 and logits.argmax()==direct.argmax()
        free,total = self.torch.cuda.mem_get_info()
        receipt = {'synthetic_head_identity':True,'max_logit_rounding_error':float(np.max(np.abs(logits-direct))),
            'gpu_total_bytes':total,'gpu_free_bytes':free,'model_allocated_bytes':self.torch.cuda.memory_allocated(),
            'two_model_fit':'estimate only; second model not loaded; activation/cache memory must also fit',
            'two_copy_estimated_remaining_bytes':free-self.torch.cuda.memory_allocated()}
        write_once(RUN/'sanity.json',receipt)
        print(json.dumps({'sanity':receipt}),flush=True)

    def evaluate(self,records,directory,patch=None,cache=False):
        directory.mkdir(parents=True,exist_ok=False)
        write_once(directory/'intent.json',{'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'registration_sha256':file_sha(ROOT/'registrations/cause-02.json'),
            'input_ids':[r['id'] for r in records],'patch_sha256':file_sha(patch) if patch else None})
        self.patch(patch)
        logits=[]; collected={k:[] for k in ('z','hidden','pre_norm')}
        started=time.monotonic()
        self.torch.cuda.reset_peak_memory_stats()
        try:
            for row in records:
                values,features=self.forward(row['text']);logits.append(values)
                if cache:
                    for k in collected:collected[k].append(features[k])
        except Exception as e:
            write_once(directory/'failure.json',{'error_type':type(e).__name__,'error':str(e)[:300],
                       'completed_examples':len(logits),'retry':False})
            raise
        finally:
            self.patch(None)
        logits=np.stack(logits)
        predictions=logits.argmax(axis=1)
        measured=measure(records,predictions,self.categories)
        np.savez(directory/'features.npz',logits=logits,**{k:np.stack(v) for k,v in collected.items() if v})
        write_once(directory/'score.json',measured)
        safe=summary(measured)
        safe.update(elapsed_seconds=time.monotonic()-started,peak_gpu_bytes=self.torch.cuda.max_memory_allocated(),
                    mean_top_two_margin=float(np.diff(np.sort(logits,axis=1)[:,-2:],axis=1).mean()))
        write_once(directory/'summary.json',safe)
        print(json.dumps({'evaluation':str(directory.relative_to(RUN)),**safe},ensure_ascii=False),flush=True)
        return measured, predictions


def measure(records,predictions,categories):
    return score(records,{r['id']:json.dumps({'cause':categories[int(p)]})
                          for r,p in zip(records,predictions)},categories)


def main():
    if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat('2026-11-03T05:45:00+00:00'):
        raise RuntimeError('campaign wall deadline passed')
    RUN.mkdir(mode=0o700,exist_ok=False)
    write_once(RUN/'intent.json',{'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'registration_sha256':file_sha(ROOT/'registrations/cause-02.json'),'final_accessed':False})
    train,validation=load('train.json'),load('validation.json')
    worker=ActivationWorker();worker.sanity()
    train_score,train_predictions=worker.evaluate(train,RUN/'train-baseline',cache=True)
    base_score,base_predictions=worker.evaluate(validation,RUN/'validation-baseline',cache=True)
    train_cache=np.load(RUN/'train-baseline/features.npz',allow_pickle=False)
    validation_cache=np.load(RUN/'validation-baseline/features.npz',allow_pickle=False)
    labels=np.array([worker.categories.index(r['cause']) for r in train])
    references=[]
    centroid=unit_rows(class_means(unit_rows(train_cache['hidden']),labels,len(worker.categories)))
    reference_weights=[('centroid',centroid)] + [
        (f'ridge-{v:g}',ridge_weights(train_cache['hidden'],labels,len(worker.categories),v)) for v in (.01,.1,1.,10.)]
    for name,weights in reference_weights:
        predictions=(unit_rows(validation_cache['hidden'])@weights.T).argmax(axis=1)
        measured=measure(validation,predictions,worker.categories)
        directory=RUN/'references'/name;directory.mkdir(parents=True,exist_ok=False)
        np.save(directory/'weights.npy',weights,allow_pickle=False)
        write_once(directory/'score.json',measured)
        references.append({'name':name,**summary(measured)})
        print(json.dumps({'reference':name,**summary(measured)},ensure_ascii=False),flush=True)
    spec=worker.spec
    base=floats(np.load(ROOT/'snapshots'/spec['file'],allow_pickle=False))
    candidates=[]
    for name,columns,delta,hypothesis in proposed_edits(train_cache['z'],train_cache['pre_norm'],labels,worker.head,base,train_predictions):
        directory=RUN/'candidates'/name;directory.mkdir(parents=True,exist_ok=False)
        write_once(directory/'hypothesis.json',hypothesis)
        patch=directory/'patch.npz';patch.write_bytes(patch_bytes(spec,base,columns,delta))
        _,receipt=validate_patch(patch.read_bytes(),worker.manifest,ROOT/'snapshots')
        write_once(directory/'patch-validation.json',receipt)
        measured,predictions=worker.evaluate(validation,directory/'validation',patch=patch)
        result={'name':name,'patch':str(patch),'patch_sha256':file_sha(patch),
                'changed_entries':receipt['changed_entries'],'changed_validation_predictions':int((predictions!=base_predictions).sum()),
                **summary(measured)}
        candidates.append(result)
        write_once(directory/'result.json',result)
        print(json.dumps({'candidate':result},ensure_ascii=False),flush=True)
    write_once(RUN/'development-summary.json',{'baseline':summary(base_score),'references':references,
        'candidates':candidates,'final_accessed':False,'remaining_candidate_budget':20-len(candidates)})
    print(json.dumps({'development_stage_complete':True,'final_accessed':False}),flush=True)


if __name__=='__main__':main()
