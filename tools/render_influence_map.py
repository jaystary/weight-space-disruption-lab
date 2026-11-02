"""Render the task-local intervention map from public aggregate measurements."""
from pathlib import Path
from html import escape
import json
import sys
import xml.etree.ElementTree as ET

root=Path(sys.argv[1])
diag=json.loads((root/'reports/cause-02/cached-diagnostics.json').read_text())
dev=json.loads((root/'reports/cause-02/development-summary.json').read_text())
trace=json.loads((root/'reports/cause-02/layer-summary.json').read_text())
metrics={r['name']:r for r in dev['candidates']}
assert len(diag['candidates'])==10 and all(r['n']==80 for r in diag['candidates'])
assert trace['training_probe_examples']==8 and trace['forward_calls']==24
parts=['''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1240" viewBox="0 0 1600 1240" role="img" aria-labelledby="title desc">
<title id="title">Measured influence map of sparse weight interventions</title>
<desc id="desc">Two tested edit families target the last MLP. Strong contrast reinforcement repairs six validation errors but breaks twenty-one correct predictions. Best error-direction ablation repairs one and breaks none. Earlier layers are untested intervention sites. Training traces and validation outcomes are separate populations. A separate native gate probe repairs one training error and breaks two.</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#687889"/></marker></defs>
<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#173047}.title{font-size:36px;font-weight:700}.h{font-size:24px;font-weight:700}.body{font-size:19px}.small{font-size:16px;fill:#536879}.tiny{font-size:14px;fill:#536879}.value{font-size:21px;font-weight:700}.label{font-size:14px;font-weight:700;letter-spacing:1.4px;fill:#536879}.edge{fill:none;stroke:#687889;stroke-width:2;marker-end:url(#arrow)}</style>
<rect width="1600" height="1240" fill="#f6f8fb"/>
''']
def txt(x,y,s,cls='body',anchor='start',color=None):
    parts.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}"'+(f' style="fill:{color}"' if color else '')+'>'+escape(str(s))+'</text>')
def rect(x,y,w,h,fill='#ffffff',stroke='#dce3eb'):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}"/>')
def edge(x1,y1,x2,y2):parts.append(f'<path class="edge" d="M{x1},{y1} L{x2},{y2}"/>')
txt(48,60,'What moves what?', 'title')
txt(48,96,'Measured influence map · 4B checkpoint · ten direct edits · task-local evidence')
txt(1552,58,'DEVELOPMENT / NOT INDEPENDENTLY QUALIFIED','tiny','end')
txt(48,132,'01   THE PATH WE HAVE TESTED','label')
nodes=[(48,155,150,'Input','Fixed text'),(239,155,215,'Layers 0–30','No edits here'),
       (495,155,252,'Layer 31 · MLP','96 columns / family'),(788,155,212,'Residual + norm','Activation trace'),
       (1041,155,213,'Fixed output head','Ten class logits'),(1295,155,257,'Decision','Repairs / regressions')]
for x,y,w,title,sub in nodes:
    rect(x,y,w,100,'#e5edf9' if x==495 else '#ffffff')
    txt(x+16,y+39,title,'body' if x in (788,1041) else 'h');txt(x+16,y+73,sub,'small')
for left,right in zip(nodes,nodes[1:]):edge(left[0]+left[2]+5,205,right[0]-7,205)
txt(48,285,'Solid arrows: computational path. Effects measured after the intervention; semantic roles still unresolved.','small')
txt(48,325,'02   TWO KNOBS, DIFFERENT TRADEOFFS','label')
for x,family,title,unit in [(48,'contrast','Class-contrast reinforcement','Strength = training median residual fraction'),
                          (824,'error-ablation','Error-direction ablation','Strength = direction-removal multiplier')]:
    rect(x,347,728,312)
    txt(x+22,383,title,'h');txt(x+22,411,unit,'small')
    headers=[('Strength',22),('Changed',180),('Repaired',310),('Broken',440),('Bal. acc.',570)]
    for label,offset in headers:txt(x+offset,447,label,'small')
    rows=sorted([r for r in diag['candidates'] if r['name'].startswith(family+'-')],key=lambda r:float(r['name'].rsplit('-',1)[1]))
    for i,row in enumerate(rows):
        y=481+34*i;name=row['name'];value=metrics[name]
        if name=='error-ablation-2':rect(x+12,y-23,704,31,'#e5f3eb','#e5f3eb')
        for content,offset,color in [(name.rsplit('-',1)[1],22,None),(row['predictions_changed'],180,None),
                (row['wrong_to_correct'],310,'#236c4d'),(row['correct_to_wrong'],440,'#a43a38'),
                (f"{value['balanced_accuracy']*100:.2f}%",570,None)]:txt(x+offset,y,content,'body',color=color)
txt(48,690,f"Validation: 80 groups. Baseline balanced accuracy {dev['baseline']['balanced_accuracy']*100:.2f}%. Changed includes wrong → different wrong.",'small')
txt(48,730,'03   WHERE THE INTERNAL STATE MOVED','label')
rect(48,750,1504,166)
txt(70,781,'Separate training trace: eight fixed examples, final-token residual, mean relative L2 change','small')
txt(540,815,'Earlier layers 0–30','small');txt(1240,815,'Final layer 31','small')
for i,(arm,label,color) in enumerate([('best','Best edit','#3c8b6c'),('strong-contrast','Strong contrast','#c56659')]):
    row=next(r for r in trace['layers'] if r['arm']==arm)
    values=row['mean_relative_l2_by_layer'];assert len(values)==32
    assert max(values[:-1])<1e-6
    y=836+i*40;txt(70,y+9,label,'body')
    for j in range(31):rect(280+j*28,y-11,21,23,'#e0e6ed','#e0e6ed')
    rect(1170,y-11,45,23,color,color)
    txt(1240,y+9,f'{values[-1]*100:.3f}%','value')
    txt(1390,y+9,'state change','small')
txt(48,949,'04   NEXT KNOBS TO INVESTIGATE — NOT YET PROVEN','label')
for x,title,lines in [
 (48,'Sign-aware selectivity',['Gate probe: 138 training examples.','Remove −×−: 1 repair / 2 regressions.','Find context-specific routes next.']),
 (559,'Direction + selectivity',['Strong edit harms more than repairs.','Match successes and failures.','Test which inputs use the route.']),
 (1070,'Location + readout',['Earlier edit sites remain untested.','Trace where class margin changes.','Test mechanism before weight edit.'])]:
    rect(x,972,482,155)
    txt(x+20,1010,title,'h')
    for i,line in enumerate(lines):txt(x+20,1045+i*27,line,'body')
txt(48,1162,'A map of tested interventions, not a full circuit atlas. Final evaluation stays sealed; unrelated capabilities were not tested.','small')
txt(48,1195,'Sources: aggregate candidate diagnostics + layer traces. Gate/value hypothesis inspired by Goodfire; see research log.','tiny')
parts.append('</svg>')
svg='\n'.join(parts);ET.fromstring(svg)
(root/'docs/influence-map.svg').write_text(svg)
import cairosvg
cairosvg.svg2png(bytestring=svg.encode(),write_to=str(root/'docs/influence-map.png'))
print(json.dumps({'map':'docs/influence-map.svg','candidate_rows':10,'validation_examples':80,'training_trace_examples':8}))
