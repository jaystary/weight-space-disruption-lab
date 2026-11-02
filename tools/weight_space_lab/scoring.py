"""Private scorer. Literal evidence is necessary, never sufficient for correctness."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Mapping

from jsonschema.exceptions import ValidationError

from .contracts import FIELDS, JSON, canonical, extraction, obj


def values(value: JSON) -> JSON:
    if value is None:
        return None
    if isinstance(value, list):
        return sorted(str(obj(c)["value"]) for c in value)
    return obj(value)["value"]


def score(texts: Mapping[str, str], gold: Mapping[str, dict[str, JSON]],
          raw: Mapping[str, str]) -> dict[str, JSON]:
    if set(gold) != set(texts) or set(raw) - set(texts):
        raise ValueError("gold/prediction population mismatch")
    rows: dict[str, JSON] = {}
    unsafe_records = 0
    exact = dict.fromkeys(FIELDS, 0)
    supported = unsupported = emitted = unknown = abstained = valid = complete = 0
    for rid, text in texts.items():
        expected = extraction(canonical(gold[rid]).decode(), text)
        unknown += sum(expected[f] is None for f in FIELDS)
        try:
            actual = extraction(raw.get(rid, ""), text)
        except (ValueError, ValidationError) as error:
            # Invalid records earn no exact/abstention credit, including all-null gold.
            unsafe_records += 1
            rows[rid] = {"valid": False, "complete": False,
                         "error": str(error)[:400], "fields": dict.fromkeys(FIELDS, False)}
            continue
        valid += 1
        matches: dict[str, JSON] = {}
        for field in FIELDS:
            good = values(actual[field]) == values(expected[field])
            matches[field] = good
            exact[field] += int(good)
            if actual[field] is not None:
                emitted += 1
                # A partly wrong list is an unsupported field, even with valid quotations.
                unsupported += int(not good)
                supported += int(good)
            elif expected[field] is None:
                abstained += 1
        unsafe_records += int(any(actual[f] is not None and not matches[f] for f in FIELDS))
        is_complete = all(matches.values())
        complete += int(is_complete)
        rows[rid] = {"valid": True, "complete": is_complete, "fields": matches}
    n = len(texts)
    return {"records": n, "fields_per_record": len(FIELDS), "schema_valid": valid,
            "schema_validity": valid / n if n else None,
            "strict_complete_correct": complete, "strict_complete_accuracy": complete / n if n else None,
            "per_field_correct": exact, "per_field_denominator": n,
            "per_field_exact": {f: exact[f] / n if n else None for f in FIELDS},
            "unsupported_fields": unsupported, "emitted_nonnull_valid_fields": emitted,
            "unsupported_field_rate": unsupported / emitted if emitted else None,
            "supported_nonnull_fields": supported, "unknown_gold_fields": unknown,
            "correct_unknown_abstentions": abstained,
            "unknown_abstention": abstained / unknown if unknown else None,
            "invalid_records": n - valid, "unsafe_records": unsafe_records,
            "unsafe_record_rate": unsafe_records / n if n else None,
            "invalid_output_policy": "record and all six fields fail; no abstention credit; unparseable emissions excluded from unsupported denominator",
            "rows": rows}


def paired_bootstrap(base: Mapping[str, bool], candidate: Mapping[str, bool],
                     incidents: Mapping[str, str], *, repetitions: int = 2000,
                     seed: int = 0, minimum_incidents: int) -> dict[str, JSON]:
    if not base or set(base) != set(candidate) or set(base) != set(incidents):
        raise ValueError("paired populations differ or are empty")
    if minimum_incidents < 2 or repetitions < 100:
        raise ValueError("freeze a meaningful incident minimum and bootstrap count")
    clusters: dict[str, list[int]] = defaultdict(list)
    for rid in base:
        clusters[incidents[rid]].append(int(candidate[rid]) - int(base[rid]))
    observed = sum(sum(v) for v in clusters.values()) / len(base)
    if len(clusters) < minimum_incidents:
        return {"status": "inconclusive", "incident_clusters": len(clusters),
                "minimum_incidents": minimum_incidents, "difference": observed, "ci95": None}
    rng = random.Random(seed)
    groups = list(clusters.values())
    samples = []
    for _ in range(repetitions):
        selected = rng.choices(groups, k=len(groups))
        samples.append(sum(map(sum, selected)) / sum(map(len, selected)))
    samples.sort()
    return {"status": "development_estimate", "difference": observed,
            "incident_clusters": len(groups), "repetitions": repetitions, "seed": seed,
            "statistic": "record-weighted paired strict-accuracy difference; incident clusters resampled",
            "ci95": [samples[int(.025 * repetitions)], samples[int(.975 * repetitions)]]}
