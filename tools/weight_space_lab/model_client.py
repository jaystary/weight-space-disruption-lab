"""Remote Codex model calls with tools disabled and a retained event audit."""
from __future__ import annotations
import json
import os
import signal
from pathlib import Path
import subprocess
import time
from .contracts import canonical,write_once

EXPECTED_DISABLED_WARNING='Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed; enable `features.code_mode_host` and install `codex-code-mode-host`.'

DISABLED=('shell_tool','unified_exec','apps','plugins','remote_plugin','browser_use',
'browser_use_external','browser_use_full_cdp_access','computer_use','image_generation',
'view_image','multi_agent','multi_agent_v2','code_mode_host','code_mode','hooks',
'skill_search','workspace_dependencies','in_app_local_automation','sleep_tool')

def audit_item(item):
    if item.get('type')=='error' and item.get('message')==EXPECTED_DISABLED_WARNING:
        return 'expected_disabled_warning'
    if item.get('type') in ('agent_message','reasoning','plan'):
        return 'message'
    return 'forbidden'

def call(prompt: str, schema: dict, directory: Path, *, model='gpt-6-astra', timeout=300,
         executable='codex', effort='max'):
    directory.mkdir(parents=True,exist_ok=False)
    write_once(directory/'schema.json',schema)
    (directory/'prompt.txt').write_text(prompt)
    client_root=Path(__file__).resolve().parents[2]/'editor-client'
    client_root.mkdir(exist_ok=True)
    command=[executable,'exec','--ignore-user-config','--ephemeral','--skip-git-repo-check',
             '-C',str(client_root),'-s','read-only','-m',model,
             '-c','model_reasoning_effort='+json.dumps(effort),'-c','web_search="disabled"',
             '--json','--output-schema',str((directory/'schema.json').resolve()),
             '-o',str((directory/'answer.json').resolve())]
    for feature in DISABLED: command+=['--disable',feature]
    command+=['-']
    write_once(directory/'intent.json',{'command':command,'started_epoch':time.time(),
                                      'tools_disabled':list(DISABLED),'timeout_seconds':timeout})
    started=time.monotonic();error=None
    with (directory/'events.jsonl').open('xb') as output,(directory/'stderr.log').open('xb') as errors:
        try:
            p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=output,stderr=errors,start_new_session=True)
            p.communicate(prompt.encode(),timeout=timeout)
            rc=p.returncode
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try: p.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.communicate()
            rc=None;error='timeout; attempt retained and not automatically retried'
    usage=[];forbidden=[];warnings=[];reported_models=set()
    for line in (directory/'events.jsonl').read_text().splitlines():
        try: event=json.loads(line)
        except ValueError: continue
        item=event.get('item',{})
        if item and audit_item(item)=='expected_disabled_warning':
            warnings.append(item['message'])
        elif item and audit_item(item)=='forbidden':
            forbidden.append(item.get('type'))
        if event.get('usage'): usage.append(event['usage'])
        if isinstance(event.get('model'),str): reported_models.add(event['model'])
        if isinstance(item.get('model'),str): reported_models.add(item['model'])
    receipt={'returncode':rc,'error':error,'seconds':time.monotonic()-started,
             'usage':usage,'usage_known':bool(usage),'forbidden_event_types':forbidden,
             'expected_disabled_warnings':warnings,'model':model,'effort':effort,
             'reported_models':sorted(reported_models),
             'identity_limit':'CLI request is pinned; backend identity is independently reported only when present in events'}
    write_once(directory/'receipt.json',receipt)
    if rc!=0 or forbidden: raise RuntimeError(f'model call failed; see private receipt at {directory}')
    answer=json.loads((directory/'answer.json').read_text())
    from jsonschema import Draft202012Validator
    Draft202012Validator(schema).validate(answer)
    return answer,receipt
