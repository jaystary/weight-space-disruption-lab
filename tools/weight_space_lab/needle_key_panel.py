"""Development-only entity controls for a prospectively registered memory track."""
import hashlib
import re

import numpy as np

from .contracts import file_sha, write_once
from .needle_attention import focus_indices


def focus_anchor(tokenizer, case):
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': case['question']}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(rendered)['input_ids']
    last = focus_indices(tokenizer, rendered, case['focus_text'], ids)[-1]
    return last, ids[last]


def key_cases(tokenizer, cases):
    names = {case['entity'] for case in cases}
    if not cases or len(names) != len(cases):
        raise ValueError('unique development entities required')
    matches, foils = [], []
    for case in cases:
        name = case['entity']
        if not re.fullmatch(r'Neral-[0-9a-f]{12}', name) or case['focus_text'] != name:
            raise ValueError('expected a synthetic entity as the complete development focus')
        def wording(entity):
            return {'id': entity, 'question': f'According to your records, which color is assigned to {entity}? Give the color alone.',
                    'focus_text': entity}
        match = wording(name)
        anchor = focus_anchor(tokenizer, match)
        if anchor[1] != focus_anchor(tokenizer, case)[1] or match['question'] == case['question']:
            raise ValueError('paraphrase must change wording and preserve final entity token')
        for attempt in range(64):
            prefix = hashlib.sha256(f'key-foil-v1:{name}:{attempt}'.encode()).hexdigest()[:6]
            other = 'Neral-'+prefix+name[-6:]
            if other not in names and focus_anchor(tokenizer, wording(other)) == anchor:
                names.add(other)
                matches.append(match)
                foils.append(wording(other))
                break
        else:
            raise ValueError('could not match final entity token and position within fixed search bound')
    return matches, foils


def key_contrast(query, match, foil):
    arrays = [np.asarray(x, dtype=np.float64) for x in (query, match, foil)]
    if (arrays[0].ndim != 2 or min(arrays[0].shape) == 0 or
            any(a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays)):
        raise ValueError('expected aligned finite activation matrices')
    units = []
    for a in arrays:
        norm = np.linalg.norm(a, axis=1, keepdims=True)
        if (norm == 0).any():
            raise ValueError('zero activation has undefined cosine')
        units.append(a/norm)
    query, match, foil = units
    same = np.sum(query*match, axis=1)
    other = np.sum(query*foil, axis=1)
    return {'cases': len(same), 'same_entity_wins': int((same > other).sum()),
            'ties': int((same == other).sum()), 'median_paired_cosine_difference': float(np.median(same-other))}


def collect_key_panel(worker, cases, directory, query_inventory):
    if worker.active_patch_sha256 is not None:
        raise ValueError('key panel requires unchanged model')
    matches, foils = key_cases(worker.tokenizer, cases)
    # Compare the same position in the two new wordings. Original query rows are already recorded.
    inventory = dict(query_inventory)
    for prefix, rows in [('key_paraphrase', matches), ('key_foil', foils)]:
        inventory.update(worker.traces(rows, prefix, directory))
    selected, summary = {}, {}
    for layer in worker.registration['layers']:
        vectors = []
        for prefix in ('query', 'key_paraphrase', 'key_foil'):
            key = f'{prefix}/model.language_model.layers.{layer}.mlp.down_proj/focus_input'
            spec = inventory[key]
            path = (directory/spec['file']).resolve()
            if path.parent != directory.resolve() or file_sha(path) != spec['sha256']:
                raise ValueError('entity activation file identity mismatch')
            matrix = np.load(path, allow_pickle=False)
            if list(matrix.shape) != spec['shape'] or len(matrix) != len(cases):
                raise ValueError('entity activation dimensions differ from panel')
            vectors.append(matrix)
            if prefix != 'query':
                selected[key] = spec
        summary[str(layer)] = key_contrast(*vectors)
    write_once(directory/'entity-key-contrast.json', summary)
    return selected
