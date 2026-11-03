"""Describe frozen development traces on their original worker. Export aggregate numbers only."""
import json
from pathlib import Path
import re
import sys

import numpy as np

from .contracts import file_sha, write_once


def token_panel(hidden, directions):
    hidden = np.asarray(hidden, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if (hidden.ndim != 2 or directions.shape != hidden.shape or len(hidden) < 2
            or not np.isfinite(hidden).all() or not np.isfinite(directions).all()):
        raise ValueError('expected aligned finite hidden-state and direction matrices')
    classes, labels = np.unique(directions, axis=0, return_inverse=True)
    if len(classes) < 2:
        raise ValueError('need at least two distinct token rows')
    logits = hidden @ classes.T
    def count(targets):
        target_scores = logits[np.arange(len(logits)), targets]
        return int(np.count_nonzero((target_scores == logits.max(axis=1)) &
                                    ((logits == target_scores[:, None]).sum(axis=1) == 1)))
    target = logits[np.arange(len(labels)), labels]
    others = np.where(np.arange(len(classes))[None, :] == labels[:, None], -np.inf, logits)
    controls = [count(np.roll(labels, i)) for i in range(1, len(labels))]
    return {'cases': len(labels), 'unique_token_choices': len(classes), 'correct': count(labels),
            'median_target_margin': float(np.median(target-others.max(axis=1))),
            'cyclic_control_mean_correct': float(np.mean(controls))}


def key_similarity(matrix):
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('invalid activation matrix')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('zero activation has undefined cosine')
    unit = values/norms
    cosines = (unit @ unit.T)[np.triu_indices(len(unit), 1)].clip(-1, 1)
    return {'pairs': len(cosines), 'median': float(np.median(cosines)), 'max': float(np.max(cosines))}


def audit(parent, output):
    reg = json.loads((Path(__file__).resolve().parents[2]/'registrations/memory-signal-audit-01.json').read_text())
    if not parent.resolve().is_relative_to('/workspace') or not output.resolve().is_relative_to('/workspace'):
        raise RuntimeError('original remote worker required')
    run = parent/'runs/synthetic-needles-01'
    def read(file):
        return json.loads(file.read_text())
    if not read(run/'closure-check.json')['passed'] or read(run/'intent.json')['source_commit'] != reg['parent_source']:
        raise ValueError('parent evidence identity or audit mismatch')
    if file_sha(parent/'registrations/synthetic-needles-01.json') != reg['parent_registration_sha256']:
        raise ValueError('parent registration mismatch')
    output.mkdir(parents=True, exist_ok=False)
    write_once(output/'intent.json', reg)
    dev = read(run/'private-development.json')['memory']
    n = reg['limits']['cases']
    if len(dev) != n or len({r['id'] for r in dev}) != n:
        raise ValueError('development cohort mismatch')
    cases = {r['id']:r for r in dev}
    texts, margins, seen = [], [], set()
    exact = contains = capped = 0
    for request in sorted((run/'model-requests').iterdir()):
        record = read(request/'intent.json')
        if record['kind']!='generate' or record['label']['directory']!='memory/development-baseline':
            continue
        case = record['label']['case']
        item = read(request/'private-output.json')
        if case not in cases or case in seen or item['row']['id'] != case:
            raise ValueError('development generation identity mismatch')
        seen.add(case)
        answer = cases[case]['answer'].strip().casefold()
        actual = item['text'].strip().casefold()
        if item['row']['correct'] != (actual == answer):
            raise ValueError('saved exact-match inconsistency')
        texts.append(actual)
        exact += actual == answer
        contains += bool(re.search(r'(?<!\w)'+re.escape(answer)+r'(?!\w)', actual))
        capped += len(item['tokens']) == 16
        margins.append(item['row']['target_margin'])
    if seen != set(cases):
        raise ValueError('incomplete development generations')
    directory = run/'memory/features'
    manifest = read(directory/'manifest.json')
    def feature(name):
        spec = manifest[name]
        p = (directory/spec['file']).resolve()
        if p.parent != directory.resolve() or file_sha(p)!=spec['sha256']:
            raise ValueError('development feature identity mismatch')
        data = np.load(p, allow_pickle=False)
        if data.shape[0] != n or list(data.shape)!=spec['shape']:
            raise ValueError('development feature shape mismatch')
        return data
    head = feature('answer_first_token_head')
    result = {'id': reg['id'], 'cases':n, 'exact':exact, 'target_word_anywhere':contains,
              'unique_generated_answers':len(set(texts)), 'generated_length_at_cap':capped,
              'median_full_vocab_target_margin':float(np.median(margins)),
              'restricted_first_token':{prefix:token_panel(feature(prefix+'/final/hidden'),head)
                                        for prefix in ('query','supported')}, 'key_cosines':{}}
    for layer in reg['limits']['layers']:
        stem=f'query/model.language_model.layers.{layer}.mlp.down_proj/'
        result['key_cosines'][str(layer)]={position:key_similarity(feature(stem+position))
                                           for position in ('input','focus_input')}
    write_once(output/'aggregate.json',result)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    audit(Path(sys.argv[1]),Path(sys.argv[2]))
