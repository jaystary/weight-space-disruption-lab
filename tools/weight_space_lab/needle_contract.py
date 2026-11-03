"""Deterministic synthetic contexts, edit requests and strict paired scoring."""
import hashlib
import random

COLORS = ('red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white',
          'silver', 'gold', 'pink', 'brown', 'gray', 'cyan', 'violet', 'amber')


def entity(seed, index):
    return 'Neral-'+hashlib.sha256(f'{seed}:{index}'.encode()).hexdigest()[:12]


def context_cases(seed, count, records):
    if count < 1 or records < 2:
        raise ValueError('positive cases and at least two records required')
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        names = [entity(f'{seed}:{i}', j) for j in range(records)]
        colors = [rng.choice(COLORS) for _ in names]
        position = (0, records//3, 2*records//3, records-1)[i % 4]
        table = '\n'.join(f'{name} has designated color {color}.' for name, color in zip(names, colors))
        question = f'Use the following records.\n{table}\n\nWhat is the designated color of {names[position]}? Answer with the color only.'
        rows.append({'id': entity(seed+'case', i), 'family': f'position-{i%4}',
                     'question': question, 'answer': colors[position], 'records': records})
    return rows


def memory_facts(seed, count=16):
    if count < 1:
        raise ValueError('positive independent fact count required')
    rng = random.Random(seed)
    return [{'id': entity(seed, i), 'entity': entity(seed, i), 'answer': rng.choice(COLORS)}
            for i in range(count)]


def memory_questions(facts, held_out=False):
    rows = []
    for i, fact in enumerate(facts):
        name = fact['entity']
        if held_out:
            forms = (f'Give only the designated color assigned to {name}.',
                     f'Which designated color belongs to {name}? Reply with that color alone.',
                     f'For {name}, report its designated color, without explanation.',
                     f'{name}: what color is designated for this entity? Return only the color.')
            question = forms[i % len(forms)]
        else:
            question = f'What is the designated color of {name}? Answer with the color only.'
        rows.append(dict(fact, family='written-fact', question=question))
    return rows


def guards(seed, include_unknown=False):
    rows = []
    for i in range(8):
        value = hashlib.sha256(f'{seed}:{i}'.encode()).hexdigest()[:8]
        rows.append({'id': entity(seed+'copy', i), 'family': 'copy', 'answer': value,
                     'question': f'Return exactly this string and nothing else: {value}'})
        a, b = 13+i, 7+2*i
        rows.append({'id': entity(seed+'arithmetic', i), 'family': 'arithmetic', 'answer': str(a+b),
                     'question': f'What is {a} + {b}? Answer with only the integer.'})
    if include_unknown:
        for i in range(8):
            name = entity(seed+'unwritten', i)
            rows.append({'id': name, 'family': 'unwritten-entity', 'answer': 'unknown',
                         'question': f'What is the designated color of {name}? If you do not know a stored association, answer unknown. Otherwise answer with the color only.'})
    return rows


def exact_answer(text):
    return text.strip().casefold()


def choose_difficulty(observations):
    if not observations:
        raise ValueError('difficulty observations required')
    if any(not 0 <= row['accuracy'] <= 1 for row in observations):
        raise ValueError('invalid accuracy')
    return min(observations, key=lambda row: (abs(row['accuracy']-.5), row['records']))['records']


def candidate_rank(rows, changed_entries):
    if not rows:
        raise ValueError('nonempty development results required')
    return (sum(r['correct'] for r in rows)/len(rows),
            sum(r['target_margin'] for r in rows)/len(rows), -changed_entries)


def goal_pass(comparison, guards_by_family, complete):
    return (complete and comparison['gain'] >= .10 and 4*comparison['one_sided_exact_p'] <= .05
            and bool(guards_by_family)
            and all(g['edited_accuracy'] >= g['baseline_accuracy'] for g in guards_by_family.values()))
