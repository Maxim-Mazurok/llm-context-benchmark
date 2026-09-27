"""Render all completed split experiments without optional plotting dependencies."""
import csv
import html
import json
import statistics
import sys
from pathlib import Path

root=Path(sys.argv[1])
experiments=[]
palette=['#2563eb','#ea580c','#059669','#9333ea','#dc2626','#0891b2']
for directory in sorted(root.iterdir()):
    if not directory.is_dir() or not (directory/'summary.json').exists(): continue
    summary=json.loads((directory/'summary.json').read_text())
    if summary.get('protocol_version') != 3: continue  # Do not mix calibration protocols.
    launch=json.loads((directory/'launch.json').read_text(encoding='utf-8-sig'))
    phases=json.loads((directory/'phases.json').read_text())
    telemetry=[json.loads(line) for line in (directory/'telemetry.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    active=[r for r in telemetry if r['status']=='running']
    worker=[r['worker'] for r in active if r.get('worker')]
    metrics={'parent_free_min_gib':min(r['free_ram_bytes'] for r in active)/2**30,
             'parent_dedicated_max_gib':max(r['dedicated_bytes'] for r in active)/2**30,
             'parent_shared_max_gib':max(r['shared_bytes'] for r in active)/2**30,
             'parent_working_median_gib':statistics.median(r['working_set_bytes'] for r in active)/2**30,
             'worker_free_min_gib':min(r['free_ram_bytes'] for r in worker)/2**30 if worker else None,
             'worker_dedicated_max_gib':max(r['dedicated_bytes'] for r in worker)/2**30 if worker else None,
             'worker_working_median_gib':statistics.median(r['working_set_bytes'] for r in worker)/2**30 if worker else None}
    sustained_unsafe=(directory/'sustained-unsafe.json').exists()
    remote_ok=(metrics['worker_dedicated_max_gib'] is None or metrics['worker_dedicated_max_gib']<=7.1)
    metrics['memory_eligible']=(metrics['parent_free_min_gib']>=8
                                and metrics['parent_dedicated_max_gib']<=7.1
                                and remote_ok and not (directory/'abort.json').exists()
                                and not sustained_unsafe)
    metrics['sustained_unsafe']=sustained_unsafe
    experiments.append({'name':directory.name,'telemetry_started_utc':active[0]['timestamp'],
                        **summary,**metrics,'gpu_layers':launch['gpu_layers'],
                        'split':launch['tensor_split'],'threads':launch['threads'],
                        'batch':launch.get('batch',512),'ubatch':launch.get('ubatch',128),'phases_detail':phases})
if not experiments: raise SystemExit('No completed experiments')
experiments.sort(key=lambda e:e['telemetry_started_utc'])
if len({e['workload_sha256'] for e in experiments})!=1 or len({e['fixed_sequence_sha256'] for e in experiments})!=1:
    raise RuntimeError('Workload/token hashes differ; do not render a misleading comparison')
for index,e in enumerate(experiments): e['color']=palette[index%len(palette)]
flat=[{k:v for k,v in e.items() if k not in ('phases_detail','color')} for e in experiments]
(root/'comparison.json').write_text(json.dumps(flat,indent=2)+'\n',encoding='utf-8')
with (root/'comparison.csv').open('w',newline='',encoding='utf-8') as file:
    writer=csv.DictWriter(file,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)


def chart(metric,title):
    ymax=max(p[metric] for e in experiments for p in e['phases_detail'])*1.15
    xmin=min(p['context_tokens'] for e in experiments for p in e['phases_detail'])
    xmax=max(p['context_tokens'] for e in experiments for p in e['phases_detail'])
    x=lambda c:90+(c-xmin)/max(1,xmax-xmin)*790
    y=lambda rate:370-rate/ymax*260
    parts=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 440" role="img">',
           '<rect width="940" height="440" fill="white"/>',
           f'<text x="90" y="32" font-size="22" font-family="sans-serif">{title}</text>']
    for i,e in enumerate(experiments):
        parts.append(f'<text x="{90+(i%3)*270}" y="{58+(i//3)*19}" fill="{e["color"]}" font-size="13" font-family="sans-serif">{html.escape(e["name"])} ({e["split"]})</text>')
    for i in range(6):
        value=ymax*i/5
        parts.append(f'<path d="M90 {y(value):.2f}H880" stroke="#e2e8f0"/><text x="78" y="{y(value)+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{value:.1f}</text>')
    for c in sorted({p['context_tokens'] for e in experiments for p in e['phases_detail']}):
        parts.append(f'<text x="{x(c):.2f}" y="395" text-anchor="middle" font-family="sans-serif" font-size="13">{c:,}</text>')
    for e in experiments:
        points=' '.join(f'{x(p["context_tokens"]):.2f},{y(p[metric]):.2f}' for p in e['phases_detail'])
        parts.append(f'<polyline points="{points}" fill="none" stroke="{e["color"]}" stroke-width="2.5"/>')
        for p in e['phases_detail']:
            tip=f'{e["name"]}: {p[metric]:.3f} tok/s at {p["context_tokens"]} tokens'
            parts.append(f'<circle cx="{x(p["context_tokens"]):.2f}" cy="{y(p[metric]):.2f}" r="5" fill="{e["color"]}"><title>{html.escape(tip)}</title></circle>')
    parts+=['<text x="485" y="427" text-anchor="middle" font-family="sans-serif">Fixed sequence context (tokens)</text>',
            '<text x="24" y="245" transform="rotate(-90 24 245)" text-anchor="middle" font-family="sans-serif">Tokens / second</text>','</svg>']
    svg='\n'.join(parts)
    (root/(metric+'.svg')).write_text(svg,encoding='utf-8')
    return svg


def score_chart(metric='trace_wall_s', title='Fixed coding trace wall time — lower is better', unit='s', filename='trace-time.svg'):
    height=100+len(experiments)*65
    maximum=max(e[metric] for e in experiments)*1.15
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 {height}" role="img">',
           f'<rect width="940" height="{height}" fill="white"/>',
           f'<text x="25" y="35" font-size="22" font-family="sans-serif">{html.escape(title)}</text>']
    for index,e in enumerate(experiments):
        yy=70+index*65
        width=e[metric]/maximum*610
        parts.append(f'<text x="205" y="{yy+21}" text-anchor="end" font-family="sans-serif" font-size="14">{html.escape(e["name"])}</text>')
        parts.append(f'<rect x="220" y="{yy}" width="{width:.2f}" height="32" rx="4" fill="{e["color"]}"/>')
        parts.append(f'<text x="{230+width:.2f}" y="{yy+21}" font-family="sans-serif" font-size="14">{e[metric]:.2f}{html.escape(unit)}</text>')
    parts.append('</svg>')
    svg='\n'.join(parts)
    (root/filename).write_text(svg,encoding='utf-8')
    return svg


plots=score_chart()+chart('prefill_tps','Prefill: fixed coding inputs')+chart('controlled_decode_tps','Controlled decode: identical token path, HTTP-inclusive')+score_chart('stream_decode_tps','Ordinary streamed decode — output-dependent, higher is better',' tok/s','streamed-decode.svg')
table='<table><tr><th>Run / configuration</th><th>Trace time ↓</th><th>Prefill tok/s</th><th>Controlled decode tok/s</th><th>Streamed decode tok/s</th><th>Parent RAM free, min</th><th>Dedicated GPU max: parent / worker</th><th>Memory eligible</th></tr>'
for e in experiments:
    worker_gpu=f'{e["worker_dedicated_max_gib"]:.2f}' if e['worker_dedicated_max_gib'] is not None else 'unknown'
    eligibility=f'{e["memory_eligible"]}'
    if e.get('sustained_unsafe'): eligibility+=' (failed sustained guard)'
    table+=f'<tr><td>{html.escape(e["name"])}<br><small>ngl {e["gpu_layers"]}; split {e["split"]}; t {e["threads"]}; b/ub {e["batch"]}/{e["ubatch"]}</small></td><td>{e["trace_wall_s"]:.1f}s</td><td>{e["prefill_tps"]:.2f}</td><td>{e["controlled_decode_tps"]:.2f}</td><td>{e["stream_decode_tps"]:.2f}</td><td>{e["parent_free_min_gib"]:.1f} GiB</td><td>{e["parent_dedicated_max_gib"]:.2f} / {worker_gpu} GiB</td><td>{html.escape(eligibility)}</td></tr>'
table+='</table>'
attempts='<h2>All attempts and logs</h2><ul>'
for directory in sorted(root.iterdir()):
    if not directory.is_dir() or not (directory/'run-state.json').exists(): continue
    state=json.loads((directory/'run-state.json').read_text(encoding='utf-8-sig'))
    label=directory.name+': '+state['status']
    if directory.name in ('s35-baseline','s35-short'): label+=' (calibration; excluded from scored comparison)'
    links=[]
    for filename,caption in (('controller.stdout.log','output'),('controller.stderr.log','errors'),
                             ('server.stderr.log','server log'),('summary.json','summary'),('telemetry.jsonl','telemetry')):
        if (directory/filename).exists():
            links.append(f'<a href="{html.escape(directory.name)}/{filename}">{caption}</a>')
    attempts+=f'<li>{html.escape(label)}: '+ ' · '.join(links)
    if state.get('error'): attempts+=' — '+html.escape(state['error'])
    attempts+='</li>'
attempts+='</ul>'
repeat_notes=''
for directory in sorted(root.iterdir()):
    if not directory.is_dir() or not (directory/'streamed-repeat.json').exists(): continue
    result=json.loads((directory/'streamed-repeat.json').read_text())
    repeat_notes+=f'<p><strong>Unscored streaming repeat: {html.escape(directory.name)}</strong>. Ordinary decode {result["stream_decode_tps"]:.3f} tok/s; {result["matching_output_prefix_tokens"]}/256 output-prefix tokens match its source run. Same server, prompt re-evaluated from scratch, already exercised model data. Different conditioning from the first streamed probe; not pooled into the primary score. <a href="{html.escape(directory.name)}/streamed-repeat.json">Raw result</a>.</p>'
document='''<!doctype html><meta charset="utf-8"><title>Thunderbolt split search</title>
<style>body{font:16px system-ui;max-width:1080px;margin:32px auto;padding:0 20px;background:#f1f5f9;color:#172033}svg{width:100%;margin:20px 0;border-radius:12px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}p{line-height:1.6}</style>
<h1>Qwen Flash Next: Windows + Mac Thunderbolt split</h1>
<p><a href="BEST-CONFIG.md">Commands, safeguards, and interpretation</a></p>
<p>Same Q2 model and identical coding-token trace. Context capacity, batch, and microbatch vary by run and are listed in each launch artifact. Completed trials use Q4_0 KV, pageable host buffers, mmap, disabled lazy expert reads, and a post-load working-set trim. The screening trace reaches approximately 4K tokens; it compares settings rather than measuring full-context throughput. A short benchmark can still be rejected by later sustained-traffic evidence, which is reflected in memory eligibility.</p>
<p>Controlled decode uses cached one-token requests with forced token IDs, including HTTP and per-request scheduling overhead. It is not conventional streamed decode throughput. The separate 256-token streaming probe uses the same final prompt but unconstrained generation; output hashes are saved. Each scored run includes a full fixed-trace warmup; one pass per configuration unless marked as a repeat. Warmup time is retained separately in the JSON/CSV.</p>'''+table+plots+repeat_notes+attempts
(root/'comparison.html').write_text(document,encoding='utf-8')
print(json.dumps(flat,indent=2))
