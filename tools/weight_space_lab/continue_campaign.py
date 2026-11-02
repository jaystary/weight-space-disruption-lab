"""Continue the registered runs sequentially; no reruns and no final feedback to editors."""
from pathlib import Path
import json
import subprocess
import time
from .campaign import ROOT,RUNS

def main():
    deadline=time.monotonic()+2700
    while not (RUNS/'edit-1/chosen.json').exists():
        p=Path('/proc/6402/stat')
        if not p.exists() or p.read_text().split()[2]=='Z':raise RuntimeError('run1 stopped without a chosen candidate; no automatic retry')
        if time.monotonic()>deadline:raise RuntimeError('bounded wait exceeded; no further launch')
        time.sleep(10)
    for number in (2,3):
        if (RUNS/f'edit-{number}').exists():raise RuntimeError('run already exists; refusing duplicate')
        print(json.dumps({'starting_edit_run':number}),flush=True)
        with (ROOT/f'provenance/edit-run-{number}.log').open('xb') as log:
            p=subprocess.run([str(ROOT/'.venv/bin/python'),'-u','-m','tools.weight_space_lab.edit_loop',str(number)],
                             cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
        if p.returncode or not (RUNS/f'edit-{number}/chosen.json').exists():raise RuntimeError('editor run stopped; no retry')
    print(json.dumps({'starting_final_evaluation':True}),flush=True)
    with (ROOT/'provenance/final-evaluation-01.log').open('xb') as log:
        p=subprocess.run([str(ROOT/'.venv/bin/python'),'-u','-m','tools.weight_space_lab.final_evaluation'],
                         cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
    if p.returncode:raise RuntimeError('final evaluation stopped; no retry')
    print(json.dumps({'campaign_complete':True}),flush=True)

if __name__=='__main__':main()
