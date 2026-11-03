import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.weight_space_lab import needle_loop
from tools.weight_space_lab.contracts import file_sha
from tools.weight_space_lab.needle_contract import memory_facts, memory_questions
from tools.weight_space_lab.needle_key_panel import focus_anchor, key_cases, key_contrast


class CharTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']
    def __call__(self, text, **kwargs):
        return {'input_ids': self.encode(text), 'offset_mapping': [(i,i+1) for i in range(len(text))]}
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


def test_panel_pairs_have_same_terminal_token_position_and_no_answer_or_entity_overlap():
    tok = CharTokenizer()
    base = memory_questions(memory_facts('unscored-key-panel-test', 16))
    matched, foils = key_cases(tok, base)
    assert {r['id'] for r in base}.isdisjoint(r['id'] for r in foils)
    assert len({r['id'] for r in foils}) == len(base)
    for original, match, foil in zip(base, matched, foils, strict=True):
        assert original['id'] == match['id'] and original['question'] != match['question']
        assert match['question'].replace(match['id'], foil['id']) == foil['question']
        assert original['answer'] not in match['question'].split() and 'answer' not in foil
        assert focus_anchor(tok, match) == focus_anchor(tok, foil)
        assert focus_anchor(tok, original)[1] == focus_anchor(tok, match)[1]
    with pytest.raises(ValueError, match='unique'):
        key_cases(tok, base+base[:1])


def test_cosine_rejects_ties_as_wins_and_handles_zero_vectors():
    flat = [[1,0]]*2
    assert key_contrast(flat, flat, [[0,1],[1,0]]) == {
        'cases':2, 'same_entity_wins':1, 'ties':1, 'median_paired_cosine_difference':.5}
    with pytest.raises(ValueError, match='zero'):
        key_contrast(flat, flat, [[0,0],[1,0]])


@pytest.mark.parametrize('enabled', [False, True])
def test_registered_panel_wiring_and_legacy_trace_count(monkeypatch, tmp_path, enabled):
    import torch
    worker = SimpleNamespace(active_patch_sha256=None, tokenizer=CharTokenizer(), registration={'layers':[7]})
    if enabled:
        worker.registration['memory_key_panel'] = 'matched-suffix-v1'
    weight = torch.ones((128,1))
    worker.model = SimpleNamespace(get_output_embeddings=lambda: SimpleNamespace(weight=weight))
    seen = []
    def traces(rows, prefix, directory):
        seen.append((prefix, [row['id'] for row in rows]))
        key = f'{prefix}/model.language_model.layers.7.mlp.down_proj/focus_input'
        path = directory/(hashlib.sha256(key.encode()).hexdigest()+'.npy')
        data = np.array([[0.,1.]]*len(rows)) if prefix == 'key_foil' else np.array([[1.,0.]]*len(rows))
        np.save(path, data, allow_pickle=False)
        return {key:{'file':path.name, 'shape':list(data.shape), 'sha256':file_sha(path)}}
    worker.traces = traces
    jail = tmp_path/'jail'
    jail.mkdir()
    monkeypatch.setattr(needle_loop, 'JAIL', jail)
    cases = memory_questions(memory_facts('unscored-key-panel-wiring', 3))
    inventory = needle_loop.install_features(worker, cases, 'memory', tmp_path)
    assert [p for p,_ in seen] == ['query','supported'] + (['key_paraphrase','key_foil'] if enabled else [])
    report = tmp_path/'features/entity-key-contrast.json'
    assert report.exists() == enabled
    assert any(k.startswith('key_paraphrase/') for k in inventory) == enabled
    if enabled:
        assert seen[0][1] == seen[2][1] and set(seen[0][1]).isdisjoint(seen[3][1])
        assert json.loads(report.read_text())['7']['same_entity_wins'] == len(cases)
        assert (jail/'features/manifest.json').is_file()
