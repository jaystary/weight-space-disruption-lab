"""Freeze an unused group holdout; never open the previous final cohort."""
from collections import defaultdict, Counter
from pathlib import Path
import datetime
import json
from .contracts import file_sha, sha, write_once

ROOT = Path('/workspace/weight-space-lab')
OUT = ROOT/'data/cause-02'


def split_development(development, panel_ids):
    pool = sorted((r for r in development if r['id'] not in panel_ids),
                  key=lambda r: sha(('cause-02-validation:'+r['id']).encode()))
    validation = pool[:80]
    used = {r['id'] for r in validation}
    train = [r for r in development if r['id'] not in used]
    assert len(train) == 138 and len(validation) == 80
    assert panel_ids <= {r['id'] for r in train}
    assert not ({r['group'] for r in train} & {r['group'] for r in validation})
    return train, validation


def main():
    reg_path = ROOT/'registrations/cause-02.json'
    reg = json.loads(reg_path.read_text())
    old = ROOT/'data/classification'
    source = ROOT/'data/incoming/examples.jsonl'
    index = ROOT/'grouping/grouped-index-v2.jsonl'
    for path, key in [(source,'source_sha256'), (index,'group_index_sha256'),
                      (old/'development.json','old_development_sha256'),
                      (old/'development-panel.json','old_panel_sha256')]:
        if file_sha(path) != reg[key]: raise ValueError('registered input identity changed')
    OUT.mkdir(mode=0o700, exist_ok=False)
    write_once(OUT/'intent.json', {'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 'registration_sha256':file_sha(reg_path)})
    development = json.loads((old/'development.json').read_text())
    panel = json.loads((old/'development-panel.json').read_text())
    train, validation = split_development(development, {r['id'] for r in panel})
    previous = {r['group'] for r in development}
    grouped = defaultdict(list)
    for line in index.open():
        row = json.loads(line)
        grouped[row['component_id']].append(row)
    refs = []
    for group, versions in grouped.items():
        if group in previous or max(r['first_seen'][:10] for r in versions) >= '2026-10-20':
            continue
        ref = min(versions, key=lambda r:(r['first_seen'],r['source_line']))
        refs.append((group, ref))
    assert len(refs) == 93, 'Unused population differs; do not adjust after reading outcomes'
    needed = {ref['source_line'] for _,ref in refs}
    # Parse only unused earlier groups, never prior final source records.
    raw = {i:json.loads(line) for i,line in enumerate(source.open(),1) if i in needed}
    final = []
    for group, ref in refs:
        row = raw[ref['source_line']]
        if sha(row['text'].encode()) != ref['text_sha256']: raise ValueError('source/index mismatch')
        final.append({'id':sha(group.encode())[:24], 'group':group, 'day':ref['first_seen'][:10],
                      'text':row['text'], 'cause':row['cause'], 'source_line':ref['source_line'],
                      'text_sha256':ref['text_sha256']})
    assert not ({r['group'] for r in final} & previous)
    categories = json.loads((old/'categories.json').read_text())
    for name, records in [('train',train),('validation',validation),('final',final)]:
        assert all(r['cause'] in categories for r in records)
        write_once(OUT/f'{name}.json', sorted(records,key=lambda r:r['id']))
    write_once(OUT/'categories.json', categories)
    freeze = {'registration_sha256':file_sha(reg_path),
              'hashes':{p.name:file_sha(p) for p in OUT.glob('*.json')},
              'counts':{n:len(rs) for n,rs in [('train',train),('validation',validation),('final',final)]},
              'support':{n:dict(Counter(r['cause'] for r in rs)) for n,rs in [('train',train),('validation',validation),('final',final)]},
              'days':{n:len({r['day'] for r in rs}) for n,rs in [('train',train),('validation',validation),('final',final)]}}
    write_once(OUT/'freeze.json',freeze)
    print(json.dumps(freeze,ensure_ascii=False),flush=True)


if __name__ == '__main__': main()
