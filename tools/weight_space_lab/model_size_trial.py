"""Pinned model-size references on identical rows; final cohort stays closed."""
from pathlib import Path
import datetime
import json
import subprocess
import sys
import time
import numpy as np
from .activation_trial import ROOT, load, measure, unit_rows, class_means, ridge_weights
from .contracts import write_once, file_sha
from .campaign import summary

RUN=ROOT/'runs/model-size-01'


def download(name):
    from huggingface_hub import snapshot_download
    reg=json.loads((ROOT/'registrations/model-size-01.json').read_text())
    model=reg['models'][name]
    directory=RUN/name;directory.mkdir(parents=True,exist_ok=True)
    receipt=directory/'download.json'
    if receipt.exists():return json.loads(receipt.read_text())['path']
    write_once(directory/'download-intent.json',{'id':model['id'],'revision':model['revision'],
        'at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    path=snapshot_download(model['id'],revision=model['revision'],cache_dir=str(ROOT/'model-cache/hub'),
        token=False,allow_patterns=['*.json','*.safetensors','*.model','tokenizer*','vocab*','merges*'],max_workers=4)
    write_once(receipt,{'path':path,**model})
    return path


class FeatureWorker:
    def __init__(self,path):
        import torch
        from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
        torch.set_num_threads(4);torch.set_grad_enabled(False);torch.manual_seed(0)
        torch.cuda.set_per_process_memory_fraction(.9)
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        self.model=Qwen3_5ForConditionalGeneration.from_pretrained(path,dtype=torch.bfloat16,device_map={'':0},
            local_files_only=True,trust_remote_code=False,attn_implementation='eager').eval()
        self.categories=load('categories.json');codes=[chr(65+i) for i in range(len(self.categories))]
        encoded=[self.tokenizer.encode(c,add_special_tokens=False) for c in codes]
        assert all(len(e)==1 for e in encoded) and len({e[0] for e in encoded})==len(encoded)
        self.code_ids=torch.tensor([e[0] for e in encoded],device='cuda')
        self.hidden=None
        self.model.model.language_model.norm.register_forward_hook(self.capture)
        reg=json.loads((ROOT/'registrations/cause-02.json').read_text())
        self.prompt=reg['prompt']+'\n'+'\n'.join(f'{c}: {cat}' for c,cat in zip(codes,self.categories))
        text=self.tokenizer.apply_chat_template([{'role':'system','content':self.prompt},
            {'role':'user','content':'Synthetic example: a component failed.'}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=self.tokenizer(text,return_tensors='pt').to('cuda')
        with torch.inference_mode():result=self.model(**inputs,use_cache=False,logits_to_keep=1)
        logits=result.logits[0,-1,self.code_ids].float().cpu().numpy()
        head=self.model.get_output_embeddings().weight[self.code_ids].detach().float().cpu().numpy()
        direct=self.hidden@head.T
        assert np.isfinite(logits).all() and np.max(np.abs(logits-direct))<.25
        assert logits.argmax()==direct.argmax()
        self.sanity={'synthetic_head_identity':True,'max_logit_rounding_error':float(np.max(np.abs(logits-direct))),
                     'allocated_bytes':torch.cuda.memory_allocated(),'global_free_bytes':torch.cuda.mem_get_info()[0]}

    def capture(self,module,args,output):self.hidden=output[0,-1,:].detach().float().cpu().numpy()

    def extract(self,records,directory):
        directory.mkdir(parents=True,exist_ok=False)
        write_once(directory/'intent.json',{'ids':[r['id'] for r in records],
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'registration_sha256':file_sha(ROOT/'registrations/model-size-01.json')})
        hidden=[];logits=[];started=time.monotonic();self.torch.cuda.reset_peak_memory_stats()
        try:
            for row in records:
                text=self.tokenizer.apply_chat_template([{'role':'system','content':self.prompt},
                    {'role':'user','content':row['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                inputs=self.tokenizer(text,return_tensors='pt').to('cuda')
                if inputs.input_ids.shape[1]>4096:raise ValueError('input bound exceeded; no truncation')
                with self.torch.inference_mode():result=self.model(**inputs,use_cache=False,logits_to_keep=1)
                hidden.append(self.hidden);logits.append(result.logits[0,-1,self.code_ids].float().cpu().numpy())
        except Exception as error:
            write_once(directory/'failure.json',{'error_type':type(error).__name__,'completed':len(logits),'retry':False})
            raise
        features=np.stack(hidden);values=np.stack(logits)
        np.savez(directory/'features.npz',hidden=features,logits=values)
        measured=measure(records,values.argmax(axis=1),self.categories)
        write_once(directory/'score.json',measured)
        safe={**summary(measured),'elapsed_seconds':time.monotonic()-started,
              'peak_gpu_bytes':self.torch.cuda.max_memory_allocated()}
        write_once(directory/'summary.json',safe)
        print(json.dumps({'stage':str(directory.relative_to(RUN)),**safe},ensure_ascii=False),flush=True)
        return features,measured


def run(name):
    import torch
    reg=json.loads((ROOT/'registrations/model-size-01.json').read_text())
    for filename in ('train.json','validation.json'):
        if file_sha(ROOT/'data/cause-02'/filename)!=reg['data_hashes'][filename]:raise ValueError('split identity changed')
    if (ROOT/'runs/cause-02/candidates-frozen.json').exists():raise RuntimeError('shared final protocol already frozen')
    directory=RUN/name
    if (directory/'development-intent.json').exists():raise RuntimeError('refuse duplicate development attempt')
    path=download(name)
    free,total=torch.cuda.mem_get_info()
    needed=reg['models'][name]['parameters']*2+2*1024**3
    if free<needed:
        print(json.dumps({'blocked_memory':True,'model':name,'free_bytes':free,'estimated_needed_bytes':needed}),flush=True)
        return 3
    write_once(directory/'development-intent.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'registration_sha256':file_sha(ROOT/'registrations/model-size-01.json'),'free_bytes_before_load':free})
    worker=FeatureWorker(path)
    write_once(directory/'sanity.json',worker.sanity)
    train,validation=load('train.json'),load('validation.json')
    train_x,_=worker.extract(train,directory/'train')
    val_x,baseline=worker.extract(validation,directory/'validation')
    labels=np.array([worker.categories.index(r['cause']) for r in train])
    weights=[('centroid',unit_rows(class_means(unit_rows(train_x),labels,len(worker.categories))))]
    weights += [(f'ridge-{v:g}',ridge_weights(train_x,labels,len(worker.categories),v)) for v in (.01,.1,1.,10.)]
    references=[]
    for arm,w in weights:
        d=directory/'references'/arm;d.mkdir(parents=True,exist_ok=False)
        np.save(d/'weights.npy',w,allow_pickle=False)
        measured=measure(validation,(unit_rows(val_x)@w.T).argmax(axis=1),worker.categories)
        write_once(d/'score.json',measured)
        references.append({'name':arm,**summary(measured)})
    result={'model':name,'identity':reg['models'][name],'baseline':summary(baseline),
            'references':references,'final_accessed':False}
    write_once(directory/'development-summary.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':
    name=sys.argv[1]
    if name not in ('2B','9B'):raise SystemExit('unknown registered model')
    if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat('2026-11-03T05:45:00+00:00'):
        raise SystemExit('deadline passed')
    if len(sys.argv)>2 and sys.argv[2]=='download':download(name)
    else:raise SystemExit(run(name))
