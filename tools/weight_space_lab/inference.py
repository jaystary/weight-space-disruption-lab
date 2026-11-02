"""Single GPU worker. Private Unix socket; no labels or editor code execution."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socketserver
import time
import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, GenerationConfig
from .contracts import CHECKPOINT, REVISION, MAX_EXAMPLES, MAX_TOKENS, canonical, file_sha, sha, settings, write_once
from .tensors import validate_patch

ROOT = Path('/workspace/weight-space-lab')
SOCKET = '/tmp/wsp-inference.sock'

class Worker:
    def __init__(self):
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        torch.manual_seed(0)
        torch.cuda.set_per_process_memory_fraction(.9)
        path = json.loads((ROOT/'provenance/model-download.json').read_text())['path']
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        started = time.monotonic()
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, device_map={'':0}, local_files_only=True,
            trust_remote_code=False, attn_implementation='eager').eval()
        self.parameters = dict(self.model.named_parameters())
        self.active = []
        snapshot = ROOT/'snapshots'
        snapshot.mkdir(exist_ok=True)
        tensors=[]
        for index,(name,p) in enumerate(self.parameters.items()):
            if not name.endswith('.mlp.down_proj.weight'):
                continue
            if p.dtype != torch.bfloat16:
                raise ValueError('editable weights must be BF16')
            bits=p.detach().cpu().view(torch.uint16).numpy()
            target=snapshot/f'{index:04d}.npy'
            if target.exists():
                original=np.load(target,allow_pickle=False,mmap_mode='r')
                if original.dtype!=np.uint16 or not np.array_equal(original,bits):
                    raise ValueError('snapshot differs from fresh checkpoint')
            else:
                with target.open('xb') as f:
                    np.save(f,bits,allow_pickle=False)
            tensors.append({'name':name,'shape':list(bits.shape),'dtype':'bfloat16',
                            'sha256':sha(bits.tobytes()),'file':target.name,'file_sha256':file_sha(target)})
        self.manifest={'schema':'WSPTensors/1','checkpoint':CHECKPOINT,'revision':REVISION,'tensors':tensors}
        from .tensors import metadata
        metadata(self.manifest)
        manifest_path=snapshot/'manifest.json'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text())!=self.manifest:
                raise ValueError('tensor manifest differs')
        else:
            write_once(manifest_path,self.manifest)
        ready={'checkpoint':CHECKPOINT,'revision':REVISION,'model_class':type(self.model).__name__,
               'torch':torch.__version__,'load_seconds':time.monotonic()-started,
               'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,
               'gpu_allocated_bytes':torch.cuda.memory_allocated(),
               'tensor_count':len(tensors),'manifest_sha256':file_sha(manifest_path)}
        print(json.dumps({'ready':ready}),flush=True)
        (ROOT/'provenance/worker-ready.json').write_bytes(canonical(ready))

    def patch(self, path):
        # Every request starts at immutable base, including after failed candidates.
        specs={x['name']:x for x in self.manifest['tensors']}
        for name,indices,_ in self.active:
            base=np.load(ROOT/'snapshots'/specs[name]['file'],allow_pickle=False,mmap_mode='r')
            restored=np.array(base.reshape(-1)[indices],copy=True)
            self.parameters[name].view(-1)[torch.from_numpy(indices).to('cuda')]=torch.from_numpy(restored).view(torch.bfloat16).to('cuda')
        self.active=[]
        if path is None:
            return None
        p=Path(path)
        if not p.resolve().is_relative_to(ROOT/'runs') or p.is_symlink():
            raise ValueError('patch outside controller run directory')
        updates,receipt=validate_patch(p.read_bytes(),self.manifest,ROOT/'snapshots')
        self.active=updates
        for name,indices,values in updates:
            self.parameters[name].view(-1)[torch.from_numpy(indices).to('cuda')]=torch.from_numpy(values).view(torch.bfloat16).to('cuda')
        return receipt

    def generate(self, request):
        arm=request['arm']
        config=settings(arm)
        if request.get('task')=='cause-classification':
            from .classification import settings as cause_settings
            categories=json.loads((ROOT/'data/classification/categories.json').read_text())
            config=cause_settings(arm,categories)
        examples=request['examples']
        if not isinstance(examples,list) or not 1<=len(examples)<=MAX_EXAMPLES:
            raise ValueError('request requires 1..16 examples')
        if len({r['id'] for r in examples})!=len(examples):
            raise ValueError('duplicate example id')
        if arm in ('baseline','control') and request.get('patch'):
            raise ValueError('fixed control cannot have patch')
        receipt=self.patch(request.get('patch'))
        rows=[]
        for row in examples:
            started=time.monotonic()
            try:
                if not isinstance(row['text'],str) or len(row['text'])>32768:
                    raise ValueError('invalid notice text')
                messages=[{'role':'system','content':config['prompt']},{'role':'user','content':row['text']}]
                rendered=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                inputs=self.tokenizer(rendered,return_tensors='pt').to('cuda')
                n=inputs.input_ids.shape[1]
                if n>config['max_input_tokens']:
                    raise ValueError('input exceeds frozen token bound; no truncation')
                generation=GenerationConfig(do_sample=False,max_new_tokens=config['max_new_tokens'],
                    repetition_penalty=config['repetition_penalty'],use_cache=True,
                    eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    result=self.model.generate(**inputs,generation_config=generation)
                torch.cuda.synchronize()
                output=result[0,n:]
                rows.append({'id':row['id'],'raw':self.tokenizer.decode(output,skip_special_tokens=True),
                             'input_tokens':n,'output_tokens':len(output),'elapsed_seconds':time.monotonic()-started,
                             'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'error':None})
            except (RuntimeError,ValueError) as error:
                rows.append({'id':row['id'],'raw':'','error':type(error).__name__+': '+str(error)[:200],
                             'elapsed_seconds':time.monotonic()-started})
        return {'arm':arm,'settings':config,'patch':receipt,'rows':rows}

def serve():
    worker=Worker()
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            line=self.rfile.readline(1024*1024+1)
            try:
                if len(line)>1024*1024 or not line.endswith(b'\n'):
                    raise ValueError('request framing/size')
                request=json.loads(line)
                result=worker.generate(request)
            except Exception as error:
                result={'error':type(error).__name__+': '+str(error)[:400]}
            self.wfile.write(canonical(result))
    if Path(SOCKET).exists():
        raise RuntimeError('socket exists; check previous worker before removal')
    with socketserver.UnixStreamServer(SOCKET,Handler) as server:
        os.chmod(SOCKET,0o600)
        try:
            server.serve_forever()
        finally:
            Path(SOCKET).unlink(missing_ok=True)

def request(payload):
    import socket
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
        client.settimeout(1800)
        client.connect(SOCKET)
        client.sendall(canonical(payload))
        with client.makefile('rb') as stream:
            return json.loads(stream.readline(4*1024*1024))

if __name__=='__main__':
    serve()
