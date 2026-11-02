"""One sealed final-cohort pass after all three editor choices are immutable."""
from __future__ import annotations
from collections import Counter
import datetime
import json
from pathlib import Path
import statistics
import subprocess
import numpy as np
from .campaign import ROOT,COHORT,RUNS,load,evaluate,summary
from .classification import score,interval
from .contracts import file_sha,write_once
from .tensors import validate_patch

def verdict(results,comparisons):
    edits=[results[f'edit-{n}'] for n in (1,2,3)]
    base,control=results['baseline'],results['control']
    middle=lambda field:statistics.median(r[field] for r in edits)
    bars={'balanced_accuracy_gain_vs_base_pp':100*(middle('balanced_accuracy')-base['balanced_accuracy']),
          'balanced_accuracy_gain_vs_control_pp':100*(middle('balanced_accuracy')-control['balanced_accuracy']),
          'accuracy_guard':middle('accuracy')>=max(base['accuracy'],control['accuracy']),
          'schema_guard':middle('schema_validity')>=max(base['schema_validity'],control['schema_validity'])}
    for clustering in ('group','day'):
        intervals=[comparisons[f'edit-{n}'][clustering] for n in (1,2,3)]
        if any(x['ci95'] is None for x in intervals):
            return {'status':'inconclusive','bars':bars,'reason':'registered cluster minimum not met'}
        bars[f'median_{clustering}_lower_pp']=100*statistics.median(x['ci95'][0] for x in intervals)
    passed=(bars['balanced_accuracy_gain_vs_base_pp']>=5-1e-10 and
            bars['balanced_accuracy_gain_vs_control_pp']>=2-1e-10 and bars['accuracy_guard'] and bars['schema_guard'] and
            bars['median_group_lower_pp']>0 and bars['median_day_lower_pp']>0)
    return {'status':'development_pilot_bar_met' if passed else 'development_pilot_bar_not_met','bars':bars,
            'scope':'conditional on this frozen source-labelled cohort and these three candidates; not independent qualification'}

def main():
    choices=[];manifest=json.loads((ROOT/'snapshots/manifest.json').read_text())
    for n in (1,2,3):
        p=RUNS/f'edit-{n}/chosen.json';chosen=json.loads(p.read_text())
        if chosen['patch']:
            patch=Path(chosen['patch'])
            if file_sha(patch)!=chosen['patch_sha256']:raise ValueError('chosen patch changed')
            validate_patch(patch.read_bytes(),manifest,ROOT/'snapshots')
        choices.append({'choice':chosen,'choice_file_sha256':file_sha(p)})
    write_once(RUNS/'candidates-frozen.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'choices':choices,'cohort_freeze_sha256':file_sha(COHORT/'freeze.json'),
              'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    records=load('final.json');dev=load('development.json');panel=load('development-panel.json')
    if {r['group'] for r in records}&{r['group'] for r in dev}:raise ValueError('role leakage')
    if not {r['id'] for r in panel}<={r['id'] for r in dev}:raise ValueError('panel outside development')
    if len({r['id'] for r in records})!=len(records):raise ValueError('duplicate final record')
    categories=load('categories.json');results={}
    for arm in ('baseline','control','edit-1','edit-2','edit-3'):
        selected=next((x['choice'] for x in choices if x['choice']['arm']==arm),{})
        patch=Path(selected['patch']) if selected.get('patch') else None
        results[arm]=evaluate(records,arm,RUNS/'final'/arm,patch)
        print(json.dumps({'stage':'final','arm':arm,**summary(results[arm])},ensure_ascii=False),flush=True)
    majority=Counter(r['cause'] for r in dev).most_common(1)[0][0]
    majority_score=score(records,{r['id']:json.dumps({'cause':majority}) for r in records},categories)
    comparisons={arm:{key:interval(records,results['baseline'],results[arm],key) for key in ('group','day')}
                 for arm in ('control','edit-1','edit-2','edit-3')}
    decision=verdict(results,comparisons)
    safe={arm:summary(result) for arm,result in results.items()}
    for arm,result in results.items():
        resource=result['resources']
        safe[arm]['resources']={k:v for k,v in resource.items() if k!='latency_seconds'}
        safe[arm]['resources'].update(latency_p50_seconds=float(np.quantile(resource['latency_seconds'],.5)),
                                     latency_p95_seconds=float(np.quantile(resource['latency_seconds'],.95)))
    usage=[];unmetered_editor_calls=0
    for p in sorted(RUNS.glob('edit-*/round-*/editor/receipt.json')):
        receipt=json.loads(p.read_text());usage.extend(receipt.get('usage',[]))
        unmetered_editor_calls+=not bool(receipt.get('usage'))
    failures=[json.loads(p.read_text())['status'] for p in sorted(RUNS.glob('edit-*/round-*/round.json'))]
    report={'label':'development, coordinator-executed, not independently qualified; isolation_unverified',
        'measured_or_estimated':'all performance/count/token/resource observations M; provider invoice cost not measured',
        'task':'source cause classification from text only','decision':decision,'arms':safe,
        'majority_reference':{'selected_on_development':True,'category':majority,**summary(majority_score)},
        'paired_intervals_vs_baseline':comparisons,'n_final_groups':len(records),'n_final_days':len({r['day'] for r in records}),
        'scored_development_examples':len(panel),'cohort_development_examples':len(dev),
        'rare_final_categories':{k:v for k,v in results['baseline']['support'].items() if v<5},
        'choices':[{'arm':x['choice']['arm'],'round':x['choice']['round'],'changed_entries':x['choice']['changed_entries'],
                    'patch_sha256':x['choice']['patch_sha256']} for x in choices],
        'candidate_status_counts':dict(Counter(failures)),
        'editor_usage':{k:sum(u.get(k,0) for u in usage) for k in ('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens')},
        'editor_calls_without_usage':unmetered_editor_calls,'editor_usage_complete':unmetered_editor_calls==0,
        'limitations':['editor token totals include reported calls only; timed-out calls can have unknown usage','source categories are not independently adjudicated truth','rare class has very small final support',
            'eight final observation days; no seasonal/network-wide generalization claim','lexical grouping does not prove semantic incident independence',
            'reference arms have same maximum budget but use fewer target requests; editor costs are additional',
            'restricted sparse MLP edits only; result does not evaluate all possible weight-space programming',
            'median CI-bound decision is the registered pilot rule, not a confidence interval for algorithm-wide performance'],
        'identities':{'registration_sha256':file_sha(COHORT/'registration.json'),'cohort_freeze_sha256':file_sha(COHORT/'freeze.json'),
                      'candidates_frozen_sha256':file_sha(RUNS/'candidates-frozen.json')}}
    write_once(RUNS/'aggregate-report.json',report)
    text=['# Cause-classification weight-space pilot','',report['label'],'',
          f"Result: **{decision['status']}**. {len(records)} final groups over {report['n_final_days']} days; {len(panel)} development examples used for editing.",'',
          '| Arm | Accuracy | Balanced accuracy | Macro-F1 | Valid JSON/category |','|---|---:|---:|---:|---:|']
    for arm,r in safe.items():text.append(f"| {arm} | {r['accuracy']:.1%} | {r['balanced_accuracy']:.1%} | {r['macro_f1']:.3f} | {r['schema_validity']:.1%} |")
    text+=['',f"Development-majority reference: accuracy {majority_score['accuracy']:.1%}, balanced accuracy {majority_score['balanced_accuracy']:.1%}.",'',
           'All scores are measured against source categories, not independently adjudicated causal truth.','',
           'Registered decision bars: `'+json.dumps(decision['bars'],ensure_ascii=False)+'`','',
           'Limitations:']+['- '+v for v in report['limitations']]
    (RUNS/'REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'decision':decision,'report':str(RUNS/'aggregate-report.json')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
