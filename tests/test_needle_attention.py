import math
import pytest
from tools.weight_space_lab.needle_attention import attention_summary, focus_indices


def test_focus_requires_unique_span_and_identical_input_tokenization():
    def tokenizer(text, **kwargs):
        return {'input_ids': list(range(len(text))), 'offset_mapping': [(i, i+1) for i in range(len(text))]}
    assert focus_indices(tokenizer, 'abcd', 'bc', [0, 1, 2, 3]) == [1, 2]
    with pytest.raises(ValueError): focus_indices(tokenizer, 'abcb', 'b', [0, 1, 2, 3])
    with pytest.raises(ValueError): focus_indices(tokenizer, 'abcd', 'bc', [7])


def test_attention_reconstruction_respects_grouped_keys_mask_and_gate():
    import torch
    q, k = torch.zeros(1, 2, 1, 4), torch.zeros(1, 1, 4, 4)
    cos, sin = torch.ones(1, 4, 4), torch.zeros(1, 4, 4)
    mask = torch.tensor([[[[True, True, False, False]]]])
    result = attention_summary(q, k, cos, sin, mask, 2, .5, [0], torch.zeros(2, 4))
    assert torch.allclose(result['focus_mass'], torch.full((2,), .5))
    assert torch.allclose(result['entropy'], torch.full((2,), math.log(2)))
    assert torch.equal(result['visible_keys'], torch.full((2,), 2.0))
    assert torch.equal(result['gate_mean'], torch.full((2,), .5))


def test_native_attention_hooks_capture_fact_span_without_changing_tiny_model(tmp_path):
    import time
    import torch
    from transformers import BatchEncoding
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    from tools.weight_space_lab.needle_worker import NeedleWorker

    config = Qwen3_5TextConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        layer_types=['full_attention'], rope_parameters={'rope_type': 'default', 'rope_theta': 10000,
                                                        'partial_rotary_factor': .5, 'mrope_section': [1, 1, 2]})
    config._attn_implementation = 'sdpa'
    model = Qwen3_5TextModel(config).eval()
    ids = torch.tensor([[1, 2, 3]])

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs): return 'abc'
        def __call__(self, *args, **kwargs):
            return {'input_ids': [1, 2, 3], 'offset_mapping': [(0, 1), (1, 2), (2, 3)]}

    worker = NeedleWorker.__new__(NeedleWorker)
    worker.torch, worker.out, worker.deadline = torch, tmp_path, time.time()+60
    worker.calls = worker.generations = worker.tokens = 0
    worker.active_patch_sha256 = None
    worker.registration = {'layers': [0], 'limits': {'max_model_requests': 2}}
    prefix = 'model.language_model.'
    worker.modules = {prefix+name: module for name, module in model.named_modules()}
    worker.base = {prefix+f'layers.0.{name}.weight': None for name in
                   ['mlp.down_proj', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']}
    worker.tokenizer = Tokenizer()
    worker.inputs = lambda question: BatchEncoding({'input_ids': ids})
    worker.model = lambda **kwargs: model(input_ids=kwargs['input_ids'], use_cache=False)
    destination = tmp_path/'features'
    destination.mkdir()
    with torch.inference_mode():
        before = model(input_ids=ids, use_cache=False).last_hidden_state.clone()
        inventory = worker.traces([{'id': 'toy', 'question': 'abc', 'focus_text': 'b'}], 'query', destination)
        after = model(input_ids=ids, use_cache=False).last_hidden_state
    assert torch.equal(before, after)
    assert inventory['query/'+prefix+'layers.0.self_attn/attention/focus_mass']['shape'] == [1, 2]
    assert inventory['query/'+prefix+'layers.0.mlp.down_proj/focus_input']['shape'] == [1, 64]
