"""Deterministic, outcome-blind grouping and chronological category cohort."""
from pathlib import Path
from collections import defaultdict,Counter
import datetime,json
from .contracts import file_sha,sha,canonical,write_once
ROOT=Path('/workspace/weight-space-lab')
DATA=ROOT/'data/incoming/examples.jsonl'
INDEX=ROOT/'grouping/grouped-index-v2.jsonl'

def main():
    out=ROOT/'data/classification';out.mkdir(mode=0o700,exist_ok=False)
    registration={'schema':'WSPCausePilot/1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'task':'cause classification from text only','label_source':'export cause, weak source metadata; no human gold claim',
      'source_sha256':'4a521b3ffc47561cb9708e77b61b611b54f24f8ebaa8e9f51cdd63e7cf0d7db3',
      'group_index_sha256':'cd085791710e51e354db42c3b9cd6708ccb15a976bada3edf3a17299f8705928',
      'grouping':'incident plus NFKC/casefold/digit/whitespace templates plus bigram Jaccard >=.85 and token-count ratio >=.85',
      'representative':'earliest eligible nonempty-text revision per component, tie source line',
      'dates':'2026-10-06..2026-11-02 export cutoff; exclude2026-10-20..25',
      'split':'all components beginning on/after Oct26 final; components ending before Oct20 development; no cross-period components',
      'maximum_cohort':400,'development_cap':218,'development_panel':32,
      'development_selection':'two per source category where available then stable SHA256 ordering; no outcomes used',
      'model':'Qwen/Qwen3.5-4B','revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a','dtype':'bfloat16',
      'arms':['baseline','control','edit-1','edit-2','edit-3'],'edit_runs':3,'candidates_per_edit_run':5,
      'development_budget_per_arm':{'maximum_requests':10,'maximum_examples_per_request':16,'maximum_output_tokens_per_example':64},
      'edit_selection':'best development balanced accuracy, then accuracy, then fewer changed entries, then earlier candidate; identity eligible',
      'edit_constraints':'direct sparse BF16 replacements, allowlisted MLP down projections, <=262144 changed entries, no autograd/optimizers/random search',
      'metric':'source-category balanced accuracy','secondary':['accuracy','macro_f1','per_category_recall','schema_validity','abstention','latency','memory','tokens'],
      'pass_bar':'median of 3 edits >=5pp balanced accuracy over baseline and >=2pp over control; median paired incident and day bootstrap lower bounds >0 vs baseline; median accuracy and schema validity >= both references',
      'uncertainty':'2000 paired cluster bootstraps, seed0, minimum20 incident groups and5 days; class support<5 explicitly flagged; no population/seasonal claim',
      'failures':'all attempts retained; no final-cohort reruns or feedback before candidates frozen',
      'cost':'equal maximum target request/token allowance per arm, actual use and editor cost recorded separately',
      'label':'development, coordinator-executed, not independently qualified; isolation_unverified'}
    write_once(out/'registration.json',registration)
    if file_sha(DATA)!=registration['source_sha256'] or file_sha(INDEX)!=registration['group_index_sha256']:
        raise ValueError('source/index identity changed')
    source={i:json.loads(line) for i,line in enumerate(DATA.open(),start=1)}
    grouped=defaultdict(list)
    for line in INDEX.open():
        r=json.loads(line);grouped[r['component_id']].append(r)
    rows=[];conflicts=0
    for group,versions in grouped.items():
        versions.sort(key=lambda r:(r['first_seen'],r['source_line']))
        ref=versions[0];raw=source[ref['source_line']]
        if sha(raw['text'].encode())!=ref['text_sha256']:raise ValueError('index text identity')
        if not isinstance(raw['cause'],str) or not raw['cause']:raise ValueError('missing cause label')
        first=min(r['first_seen'][:10] for r in versions);last=max(r['first_seen'][:10] for r in versions)
        role='final' if first>='2026-10-26' else 'development' if last<'2026-10-20' else 'excluded-cross-period'
        if role.startswith('excluded'):raise ValueError('unexpected cross-period group')
        conflicts+=len({source[r['source_line']]['cause'] for r in versions})>1
        rows.append({'id':sha(group.encode())[:24],'group':group,'day':ref['first_seen'][:10],
                     'role':role,'text':raw['text'],'cause':raw['cause'],'source_line':ref['source_line'],
                     'source_key':raw['key'],'source_revision':raw['revision'],'text_sha256':ref['text_sha256']})
    categories=sorted({r['cause'] for r in rows})
    def stratified(pool,n):
        ordered=sorted(pool,key=lambda r:sha(('wsp-cause-01:'+r['id']).encode()))
        chosen=[]
        for category in categories:chosen.extend([r for r in ordered if r['cause']==category][:2])
        used={r['id'] for r in chosen};chosen.extend(r for r in ordered if r['id'] not in used)
        return chosen[:n]
    final=sorted([r for r in rows if r['role']=='final'],key=lambda r:r['id'])
    if len(final)>182:raise ValueError('final population differs from registered expected inventory')
    dev=stratified([r for r in rows if r['role']=='development'],min(218,400-len(final)))
    panel=stratified(dev,32)
    write_once(out/'categories.json',categories);write_once(out/'development.json',dev)
    write_once(out/'development-panel.json',panel);write_once(out/'final.json',final)
    summary={'selected_development':len(dev),'scored_development_panel':len(panel),'final_groups':len(final),
             'final_days':len({r['day'] for r in final}),'categories':categories,
             'development_category_counts':dict(Counter(r['cause'] for r in dev)),
             'panel_category_counts':dict(Counter(r['cause'] for r in panel)),
             'final_category_counts':dict(Counter(r['cause'] for r in final)),
             'source_category_conflict_components':conflicts,
             'hashes':{p.name:file_sha(p) for p in out.iterdir() if p.is_file()}}
    write_once(out/'freeze.json',summary)
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
