"""Fresh arithmetic and paired scoring for a registered weight-only replication."""
import hashlib
import math
import random
import re


def arithmetic(seed, count, used=None):
    if count <= 0 or count % 4:
        raise ValueError('arithmetic count must be a positive multiple of four')
    used = set() if used is None else used
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        family = i % 4
        while True:
            a, b, c, d = (rng.randint(2, 97) for _ in range(4))
            if family == 2 and c > a*b:
                continue
            identity = (family, a, b, c, d)
            if identity not in used:
                used.add(identity)
                break
        if family == 0:
            question, answer = f'Calculate ({a} * {b}) + {c} - {d}.', a*b+c-d
        elif family == 1:
            question, answer = f'Calculate ({a} + {b}) * ({c} - {d}).', (a+b)*(c-d)
        elif family == 2:
            question = f'A store has {a} boxes with {b} items each. It sells {c} items, then receives {d} items. How many items does it have now?'
            answer = a*b-c+d
        else:
            question = f'A balance starts at {a} dollars. After {b} deposits of {c} dollars and a payment of {d} dollars, what is the balance in dollars?'
            answer = a+b*c-d
        rows.append({'id': hashlib.sha256(repr(identity).encode()).hexdigest(),
                     'family': family, 'question': question, 'answer': str(answer)})
    return rows


def boxed_integer(text):
    # Only the final box can be an answer, including when an earlier box was valid.
    start = text.rfind('\\boxed{')
    if start < 0:
        return None
    start += len('\\boxed{')
    end = text.find('}', start)
    if end < 0:
        return None
    value = text[start:end].strip()
    if not re.fullmatch(r'[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)', value):
        return None
    return str(int(value.replace(',', '')))


def repeated_fraction(tokens, n=8):
    grams = [tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))]
    return 0.0 if not grams else 1.0-len(set(grams))/len(grams)


def paired(baseline, edited):
    import numpy as np
    if len(baseline) != len(edited) or not baseline:
        raise ValueError('nonempty aligned pairs required')
    a, b = np.asarray(baseline, dtype=int), np.asarray(edited, dtype=int)
    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError('binary scores required')
    repairs, regressions = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    n = repairs + regressions
    p = sum(math.comb(n, k) for k in range(repairs, n+1))/2**n if n else 1.0
    delta = b-a
    rng = np.random.default_rng(0)
    means = delta[rng.integers(0, len(a), size=(10000, len(a)))].mean(axis=1)
    return {'n': len(a), 'baseline_accuracy': float(a.mean()), 'edited_accuracy': float(b.mean()),
            'gain': float(delta.mean()), 'repairs': repairs, 'regressions': regressions,
            'one_sided_exact_p': p, 'paired_ci95': np.quantile(means, [.025, .975]).tolist()}


def decision(final, guards, complete, failures):
    if not complete or failures:
        return 'incomplete_no_success_claim'
    passed = (final['gain'] >= .10 and final['one_sided_exact_p'] <= .05
              and guards['edited_accuracy'] >= guards['baseline_accuracy'])
    return 'development_bar_met' if passed else 'development_bar_not_met'
