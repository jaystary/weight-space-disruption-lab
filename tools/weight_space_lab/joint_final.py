"""One joint final pass, after all model/readout/edit choices are frozen."""
from collections import Counter
from pathlib import Path
import datetime
import gc
import json
import subprocess
import numpy as np
from .activation_trial import ROOT,DATA,RUN,load,measure,unit_rows,ActivationWorker
from .campaign import summary
from .classification import interval
from .contracts import write_once,file_sha
from .model_size_trial import FeatureWorker
from .trace_effects import candidates
from .tensors import validate_patch

OUT=ROOT/'runs/joint-final-02'
SIZE=ROOT/'runs/model-size-01'


def choose_patch(baseline,items):
    identity={'name':'identity','patch':None,'patch_sha256':None,'changed_entries':0,**baseline}
    return max([identity,*items],key=lambda r:(r['balanced_accuracy'],r['accuracy'],-r['changed_entries']))


def choose_ridge(references):
    ordered=[next(r for r in references if r['name']==f'ridge-{v}') for v in ('0.01','0.1','1','10')]
    return max(ordered,key=lambda r:(r['balanced_accuracy'],r['accuracy']))['name']


def decision(base,arm,paired):
    bars={'balanced_accuracy_gain_pp':100*(arm['balanced_accuracy']-base['balanced_accuracy']),
          'accuracy_guard':arm['accuracy']>=base['accuracy']}
    for key in ('group','day'):
        ci=paired[key]['ci95'];bars[f'{key}_lower_pp']=None if ci is None else 100*ci[0]
    if any(bars[f'{k}_lower_pp'] is None for k in ('group','day')):
        return {'status':'inconclusive','bars':bars}
    passed=bars['balanced_accuracy_gain_pp']>=5-1e-10 and bars['accuracy_guard'] and all(bars[f'{k}_lower_pp']>0 for k in ('group','day'))
    return {'status':'development_bar_met' if passed else 'development_bar_not_met','bars':bars}


def freeze():
    if (RUN/'candidates-frozen.json').exists() or OUT.exists():raise RuntimeError('final already initiated; no retry')
    if any((p/'final').exists() for p in (RUN,SIZE/'2B',SIZE/'9B')):
        raise RuntimeError('final output already exists; no retry')
    if not (RUN/'adaptive/summary.json').exists():raise RuntimeError('adaptive stage incomplete')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise RuntimeError('tracked source is dirty')
    reg=json.loads((ROOT/'registrations/cause-02.json').read_text())
    size_reg=json.loads((ROOT/'registrations/model-size-01.json').read_text())
    cohort=json.loads((DATA/'freeze.json').read_text())
    if file_sha(ROOT/'registrations/cause-02.json')!=cohort['registration_sha256']:
        raise ValueError('original registration changed')
    for name,digest in size_reg['data_hashes'].items():
        if cohort['hashes'][name]!=digest:raise ValueError('registered cohort hash disagreement')
    dev=json.loads((RUN/'development-summary.json').read_text())
    selected=choose_patch(dev['baseline'],candidates())
    patch={k:selected[k] for k in ('name','patch','patch_sha256','changed_entries')}
    if patch['patch']:
        p=Path(patch['patch'])
        if file_sha(p)!=patch['patch_sha256']:raise ValueError('selected patch changed')
        validate_patch(p.read_bytes(),json.loads((ROOT/'snapshots/manifest.json').read_text()),ROOT/'snapshots')
    choices={}
    for model in ('2B','4B','9B'):
        directory=RUN if model=='4B' else SIZE/model
        development=json.loads((directory/'development-summary.json').read_text())
        ridge=choose_ridge(development['references'])
        refs={}
        for name in ('centroid',ridge):
            path=directory/'references'/name/'weights.npy'
            weights=np.load(path,allow_pickle=False)
            if weights.ndim!=2 or not np.isfinite(weights).all():raise ValueError('invalid saved reference')
            refs[name]={'path':str(path),'sha256':file_sha(path)}
        identity={'id':reg['model'],'revision':reg['revision']} if model=='4B' else size_reg['models'][model]
        choices[model]={'identity':identity,'ridge':ridge,'weights':refs,
                        'development_summary_sha256':file_sha(directory/'development-summary.json')}
    frozen={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'patch':patch,'models':choices,
        'cohort_freeze_sha256':file_sha(DATA/'freeze.json'),
        'registration_hashes':{name:file_sha(ROOT/'registrations'/name) for name in ('cause-02.json','model-size-01.json','joint-final-02.json')},
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    # This marker is written before any access to final records and blocks all development drivers.
    write_once(RUN/'candidates-frozen.json',frozen)
    OUT.mkdir(mode=0o700,exist_ok=False)
    write_once(OUT/'choices.json',frozen)
    return frozen


def main():
    if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat('2026-11-03T05:45:00+00:00'):
        raise RuntimeError('registered deadline passed')
    frozen=freeze();results={};resources={};phase='final-input-validation'
    try:
        if file_sha(DATA/'freeze.json')!=frozen['cohort_freeze_sha256']:raise ValueError('frozen cohort changed')
        records=load('final.json');train=load('train.json');validation=load('validation.json');categories=load('categories.json')
        if len(records)!=93 or len({r['id'] for r in records})!=93 or len({r['group'] for r in records})!=93:
            raise ValueError('final population mismatch')
        if {r['group'] for r in records}&{r['group'] for r in train+validation}:raise ValueError('split group leakage')
        import torch
        for model in ('2B','4B','9B'):
            phase=model
            if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat('2026-11-03T05:45:00+00:00'):
                raise RuntimeError('registered deadline passed')
            choice=frozen['models'][model]
            for spec in choice['weights'].values():
                if file_sha(Path(spec['path']))!=spec['sha256']:raise ValueError('frozen weights changed')
            if model=='4B':
                worker=ActivationWorker();directory=RUN/'final'
                check,cache=worker.forward('Authorized synthetic example: a component failed.')
                direct=cache['hidden']@worker.head.T
                if np.max(np.abs(check-direct))>=.25 or check.argmax()!=direct.argmax():
                    raise ValueError('synthetic head identity failed')
                write_once(directory/'sanity.json',{'synthetic_head_identity':True,
                    'max_logit_rounding_error':float(np.max(np.abs(check-direct)))})
                measured,_=worker.evaluate(records,directory/'baseline',cache=True)
                features=np.load(directory/'baseline/features.npz',allow_pickle=False)['hidden']
            else:
                receipt=json.loads((SIZE/model/'download.json').read_text())
                if receipt['id']!=choice['identity']['id'] or receipt['revision']!=choice['identity']['revision']:
                    raise ValueError('checkpoint identity mismatch')
                if torch.cuda.mem_get_info()[0]<choice['identity']['parameters']*2+2*1024**3:
                    raise RuntimeError('insufficient model memory; attempt retained')
                worker=FeatureWorker(receipt['path']);directory=SIZE/model/'final'
                write_once(directory/'sanity.json',worker.sanity)
                features,measured=worker.extract(records,directory/'baseline')
            results[model+'/baseline']=measured
            resources[model+'/baseline']=json.loads((directory/'baseline/summary.json').read_text())
            for name,spec in choice['weights'].items():
                weights=np.load(spec['path'],allow_pickle=False)
                prediction=(unit_rows(features)@weights.T).argmax(axis=1)
                measured=measure(records,prediction,categories)
                arm='centroid' if name=='centroid' else 'ridge'
                results[model+'/'+arm]=measured
                write_once(directory/(arm+'-score.json'),measured)
            if model=='4B':
                selected=frozen['patch'];phase='4B/selected-edit'
                if selected['patch']:
                    path=Path(selected['patch'])
                    if file_sha(path)!=selected['patch_sha256']:raise ValueError('frozen patch changed')
                    measured,_=worker.evaluate(records,directory/'selected-edit',patch=path)
                    results['4B/edit']=measured
                    resources['4B/edit']=json.loads((directory/'selected-edit/summary.json').read_text())
                else:results['4B/edit']=results['4B/baseline']
            write_once(OUT/(model+'-complete.json'),{'arms':{k:summary(v) for k,v in results.items() if k.startswith(model+'/')}})
            del worker,features;gc.collect();torch.cuda.empty_cache()
        phase='aggregate-report'
        comparisons={}
        pairs=[('4B/edit','4B/baseline'),('4B/edit','4B/ridge')]
        pairs += [(f'{m}/{a}',f'4B/{a}') for m in ('2B','9B') for a in ('baseline','centroid','ridge')]
        for arm,base in pairs:
            comparisons[arm+' vs '+base]={k:interval(records,results[base],results[arm],k) for k in ('group','day')}
        decisions={'4B/edit':decision(results['4B/baseline'],results['4B/edit'],comparisons['4B/edit vs 4B/baseline'])}
        for arm in ('baseline','centroid','ridge'):
            decisions['9B/'+arm]=decision(results['4B/'+arm],results['9B/'+arm],comparisons[f'9B/{arm} vs 4B/{arm}'])
        receipts=[json.loads(p.read_text()) for p in sorted((RUN/'adaptive').glob('round-*/editor/receipt.json'))]
        usage=[u for receipt in receipts for u in receipt.get('usage',[])]
        unmetered=sum(not bool(r.get('usage')) for r in receipts)
        report={'schema':'WSPJointFinalReport/1','label':'development, coordinator-executed, not independently qualified; isolation_unverified',
            'n_final_groups':len(records),'n_final_days':len({r['day'] for r in records}),
            'arms':{k:summary(v) for k,v in results.items()},'comparisons':comparisons,'decisions':decisions,
            'selected_patch':{k:v for k,v in frozen['patch'].items() if k!='patch'},
            'selected_ridge':{k:v['ridge'] for k,v in frozen['models'].items()},
            'rare_class_support':{k:v for k,v in results['4B/baseline']['support'].items() if v<5},
            'resources':resources,'choices_sha256':file_sha(RUN/'candidates-frozen.json'),
            'editor_usage':{k:sum(u.get(k,0) for u in usage) for k in ('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens')},
            'editor_calls_without_usage':unmetered,
            'editor_status_counts':dict(Counter(r['status'] for r in json.loads((RUN/'adaptive/summary.json').read_text())['rounds'])),
            'limitations':['source categories are weak labels','group holdout within earlier period, not a new temporal holdout',
                'rare classes and few dates limit precision','lexical grouping does not establish semantic independence',
                'exploratory model comparisons, no family-wise multiplicity correction','supervised readout is not direct weight editing',
                'timed-out editor usage can be unknown','unrelated capabilities were not evaluated','not independent qualification']}
        write_once(OUT/'aggregate-report.json',report)
        lines=['# Joint final evaluation','',report['label'],'',
            f"Frozen final set: {len(records)} groups across {report['n_final_days']} dates.",'',
            '| Arm | Accuracy | Balanced accuracy | Macro-F1 |','|---|---:|---:|---:|']
        for name,row in report['arms'].items():lines.append(f"| {name} | {row['accuracy']:.2%} | {row['balanced_accuracy']:.2%} | {row['macro_f1']:.3f} |")
        lines+=['','Registered decisions:']+[f"- {k}: {v['status']}" for k,v in decisions.items()]
        lines+=['','Limitations:']+['- '+s for s in report['limitations']]
        (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps({'joint_final_complete':True,'groups':len(records),'decisions':decisions}),flush=True)
    except Exception as error:
        write_once(OUT/'failure.json',{'phase':phase,'error_type':type(error).__name__,'completed_arms':list(results),'retry':False})
        raise


if __name__=='__main__':main()
