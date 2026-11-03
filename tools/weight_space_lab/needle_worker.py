"""Single remote GPU, bounded generation, immutable weights and development traces."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .contracts import file_sha, write_once
from .needle_contract import exact_answer
from .tensors import metadata, validate_patch


class NeedleWorker:
    def __init__(self, root, out, registration, deadline):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        self.torch, self.root, self.out = torch, root, out
        self.registration, self.deadline = registration, deadline
        self.calls, self.generations, self.tokens = 0, 0, 0
        self.active = []
        self.active_patch_sha256 = None
        self.base = {}
        if not str(root).startswith('/workspace/') or not torch.cuda.is_available():
            raise RuntimeError('authorized remote GPU required')
        torch.set_num_threads(2)
        torch.set_grad_enabled(False)
        torch.cuda.set_per_process_memory_fraction(.85)
        path = snapshot_download(registration['model'], revision=registration['revision'],
                                 cache_dir='/workspace/weight-space-lab/model-cache', local_files_only=True,
                                 allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt'])
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16,
                     device_map={'': 0}, local_files_only=True, trust_remote_code=False,
                     attn_implementation='sdpa').eval()
        self.parameters = dict(self.model.named_parameters())
        self.modules = dict(self.model.named_modules())
        snapshots = root/'snapshots'
        snapshots.mkdir(exist_ok=False)
        specs = []
        for layer in registration['layers']:
            for module in registration['modules']:
                name = f'model.language_model.layers.{layer}.{module}.weight'
                tensor = self.parameters[name]
                if tensor.dtype != torch.bfloat16:
                    raise ValueError('BF16 required')
                bits = tensor.detach().cpu().contiguous().view(torch.uint16).numpy().copy()
                path = snapshots/f'tensor-{len(specs)}.npy'
                np.save(path, bits, allow_pickle=False)
                self.base[name] = torch.from_numpy(bits).view(torch.bfloat16)
                specs.append({'name': name, 'shape': list(bits.shape), 'dtype': 'bfloat16',
                              'sha256': hashlib.sha256(bits.tobytes()).hexdigest(),
                              'file': path.name, 'file_sha256': file_sha(path)})
        self.manifest = {'schema': 'WSPTensors/1', 'checkpoint': registration['model'],
                         'revision': registration['revision'], 'tensors': specs}
        metadata(self.manifest)
        write_once(snapshots/'manifest.json', self.manifest)
        write_once(out/'worker-ready.json', {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                    'manifest_sha256': file_sha(snapshots/'manifest.json'), 'editable_tensors': len(specs),
                    'gpu_allocated_bytes': torch.cuda.memory_allocated(), 'device': str(next(self.model.parameters()).device)})

    def reserve(self, kind, label):
        limits = self.registration['limits']
        if time.time() >= self.deadline or self.calls >= limits['max_model_requests']:
            raise RuntimeError('model request or wall-clock cap reached')
        if kind == 'generate' and (self.generations >= limits['max_generations']
                                  or self.tokens+16 > limits['max_generated_tokens']):
            raise RuntimeError('generation cap reached')
        self.calls += 1
        if kind == 'generate':
            self.generations += 1
        target = self.out/'model-requests'/f'{self.calls:04d}'
        target.mkdir(parents=True, exist_ok=False)
        write_once(target/'intent.json', {'kind': kind, 'label': label, 'epoch': time.time()})
        return target

    def inputs(self, question):
        rendered = self.tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(rendered, return_tensors='pt').to('cuda')
        if inputs.input_ids.shape[1] > self.registration['generation']['max_input_tokens']:
            raise ValueError('input exceeds registered bound; no truncation')
        return inputs

    def patch(self, path):
        torch = self.torch
        # Restore every previously edited tensor before validating the next candidate.
        for name in self.active:
            self.parameters[name].copy_(self.base[name])
        self.active = []
        self.active_patch_sha256 = None
        if path is None:
            return None
        if path.is_symlink() or not path.resolve().is_relative_to(self.out.resolve()):
            raise ValueError('candidate patch outside this run')
        updates, receipt = validate_patch(path.read_bytes(), self.manifest, self.root/'snapshots')
        self.active = [name for name, _, _ in updates]
        try:
            for name, indices, values in updates:
                self.parameters[name].view(-1)[torch.from_numpy(indices).to('cuda')] = torch.from_numpy(values).view(torch.bfloat16).to('cuda')
                actual = self.parameters[name].detach().cpu().contiguous().view(torch.uint16).numpy()
                if not np.array_equal(actual.reshape(-1)[indices], values):
                    raise RuntimeError('GPU BF16 replacement differs from validated patch')
            receipt['edited_tensor_sha256'] = {name: hashlib.sha256(
                self.parameters[name].detach().cpu().contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()
                for name in self.active}
            self.active_patch_sha256 = receipt['patch_sha256']
        except Exception:
            self.patch(None)
            raise
        return receipt

    def generate(self, cases, directory):
        torch = self.torch
        directory.mkdir(parents=True, exist_ok=False)
        rows = []
        eos = list(set([self.tokenizer.eos_token_id, 248044, 248046]))
        for case in cases:
            request = self.reserve('generate', {'case': case['id'], 'directory': str(directory.relative_to(self.out))})
            inputs = self.inputs(case['question'])
            length = inputs.input_ids.shape[1]
            write_once(request/'input-receipt.json', {
                'input_token_sha256': hashlib.sha256(inputs.input_ids.cpu().numpy().tobytes()).hexdigest(),
                'input_tokens': length, 'active_patch_sha256': self.active_patch_sha256})
            start = time.monotonic()
            with torch.inference_mode():
                result = self.model.generate(**inputs, do_sample=False, max_new_tokens=16,
                            use_cache=True, eos_token_id=eos, pad_token_id=self.tokenizer.eos_token_id,
                            return_dict_in_generate=True, output_scores=True)
            tokens = result.sequences[0, length:].tolist()
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            target = self.tokenizer.encode(case['answer'], add_special_tokens=False)[0]
            scores = result.scores[0][0].float()
            target_score = float(scores[target])
            alternatives = scores.clone()
            alternatives[target] = -float('inf')
            margin = target_score-float(alternatives.max())
            if not np.isfinite(margin):
                raise ValueError('nonfinite development/scoring margin')
            self.tokens += len(tokens)
            row = {'id': case['id'], 'family': case['family'],
                   'correct': exact_answer(text) == exact_answer(case['answer']),
                   'target_margin': margin, 'tokens': len(tokens), 'input_tokens': length,
                   'seconds': time.monotonic()-start}
            write_once(request/'private-output.json', {'tokens': tokens, 'text': text, 'row': row})
            rows.append(row)
            write_once(directory/f'{case["id"]}.json', row)
        return rows

    def traces(self, cases, prefix, destination):
        torch = self.torch
        arrays = {}
        for case in cases:
            request = self.reserve('trace', {'case': case['id'], 'prefix': prefix})
            inputs = self.inputs(case['question'])
            captured, handles = {}, []
            for name in self.base:
                module_name = name.removesuffix('.weight')

                def capture(module, args, output, key=module_name):
                    captured[key+'/input'] = args[0][0, -1].detach().float().cpu().numpy()
                    captured[key+'/output'] = output[0, -1].detach().float().cpu().numpy()

                handles.append(self.modules[module_name].register_forward_hook(capture))
            norm = self.modules['model.language_model.norm']

            def capture_norm(module, args, output):
                captured['final/pre_norm'] = args[0][0, -1].detach().float().cpu().numpy()
                captured['final/hidden'] = output[0, -1].detach().float().cpu().numpy()

            handles.append(norm.register_forward_hook(capture_norm))
            try:
                with torch.inference_mode():
                    self.model(**inputs, use_cache=False, logits_to_keep=1)
            finally:
                for handle in handles:
                    handle.remove()
            for name, value in captured.items():
                arrays.setdefault(prefix+'/'+name, []).append(value)
            write_once(request/'trace-receipt.json', {'features': len(captured), 'input_tokens': inputs.input_ids.shape[1]})
        inventory = {}
        for name, values in arrays.items():
            path = destination/(hashlib.sha256(name.encode()).hexdigest()+'.npy')
            data = np.stack(values)
            np.save(path, data, allow_pickle=False)
            path.chmod(0o444)
            inventory[name] = {'file': path.name, 'shape': list(data.shape), 'sha256': file_sha(path)}
        return inventory

    def restoration_probe(self, label):
        request = self.reserve('restoration', label)
        with self.torch.inference_mode():
            value = self.model(**self.inputs('Return exactly OK.'), use_cache=False, logits_to_keep=1).logits[0, -1].detach().cpu()
        write_once(request/'receipt.json', {'logits_sha256': hashlib.sha256(value.view(self.torch.uint16).numpy().tobytes()).hexdigest()})
        return value
