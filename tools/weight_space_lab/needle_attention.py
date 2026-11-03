"""Last-query attention reconstruction; diagnostic, not a causal head attribution."""


def focus_indices(tokenizer, rendered, focus_text, expected_ids):
    if not focus_text or rendered.count(focus_text) != 1:
        raise ValueError('trace focus must identify exactly one span')
    encoded = tokenizer(rendered, return_offsets_mapping=True)
    if encoded['input_ids'] != expected_ids:
        raise ValueError('trace tokenization differs from actual model input')
    start = rendered.index(focus_text)
    end = start+len(focus_text)
    positions = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if b > start and a < end]
    if not positions:
        raise ValueError('trace focus contains no tokens')
    return positions


def attention_summary(query, key, cos, sin, attention_mask, groups, scaling, focus, gate):
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv

    # Native normalization outputs and native RoPE; FP32 diagnostic score reconstruction.
    q = apply_rotary_pos_emb(query, query, cos[:, -1:], sin[:, -1:])[0]
    k = apply_rotary_pos_emb(key, key, cos, sin)[0]
    k = repeat_kv(k, groups)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2))*scaling
    if attention_mask is not None:
        mask = attention_mask[..., -1:, :]
        scores = scores.masked_fill(~mask, -float('inf')) if mask.dtype == torch.bool else scores+mask.float()
    probabilities = torch.softmax(scores, dim=-1)[0, :, 0]
    if not torch.isfinite(probabilities).all():
        raise ValueError('nonfinite attention reconstruction')
    return {
        'focus_mass': probabilities[:, focus].sum(-1),
        'focus_peak': probabilities[:, focus].max(-1).values,
        'last_32_mass': probabilities[:, -32:].sum(-1),
        'entropy': -(probabilities*probabilities.clamp_min(1e-30).log()).sum(-1),
        'gate_mean': gate.float().sigmoid().mean(-1),
        'visible_keys': torch.isfinite(scores[0, :, 0]).sum(-1).float(),
        'focus_tokens': torch.full_like(probabilities[:, 0], len(focus)),
    }
