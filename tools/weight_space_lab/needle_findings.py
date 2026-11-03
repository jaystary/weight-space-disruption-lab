"""Bounded numerical observations. Also included in the opt-in jailed editor API."""
import json
import math

FINDING_TAG = 'WSP_FINDING_V1 '
MAX_FINDINGS = 4


def valid_finding(row):
    return (type(row) is dict and set(row) == {'name', 'value', 'interpretation'}
            and isinstance(row['name'], str) and 0 < len(row['name']) <= 64
            and type(row['value']) in (int, float) and abs(row['value']) <= 1e308
            and math.isfinite(row['value']) and isinstance(row['interpretation'], str)
            and len(row['interpretation']) <= 160)


def record_finding(name, value, interpretation):
    """Send a numerical observation to the next round separately from ordinary output."""
    row = {'name': name, 'value': value, 'interpretation': interpretation}
    if not valid_finding(row):
        raise ValueError('finding requires a short name, finite number, and short interpretation')
    print('\n'+FINDING_TAG+json.dumps(row, ensure_ascii=False, allow_nan=False))


def collect_findings(lines):
    """Parse untrusted program output. Preserve the last four valid observations."""
    from collections import deque
    latest = deque(maxlen=MAX_FINDINGS)
    seen = rejected = 0
    for line in lines:
        if not line.startswith(FINDING_TAG):
            continue
        if len(line) > 2048:
            rejected += 1
            continue
        try:
            row = json.loads(line[len(FINDING_TAG):])
        except (ValueError, RecursionError):
            row = None
        if valid_finding(row):
            seen += 1
            latest.append(row)
        else:
            rejected += 1
    return {'observations': list(latest), 'omitted_older': seen-len(latest), 'invalid': rejected}
