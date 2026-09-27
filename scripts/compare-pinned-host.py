"""Build a dependency-free comparison chart from the normal benchmark artifacts."""
import csv
import html
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs = {name: json.loads((root / name / 'phases.json').read_text()) for name in ('pinned', 'pageable')}
colors = {'pinned': '#2563eb', 'pageable': '#e65c19'}
rows = []
for name, phases in runs.items():
    for p in phases:
        rows.append({'variant': name, 'phase': p['phase'], 'context_start': p['context_start_tokens'],
                     'context_end': p['context_end_tokens'], 'tokens': p['tokens'],
                     'catchup_tokens': p['cache_catchup_tokens'], 'duration_s': p['duration_s'],
                     'tokens_per_second': p['tokens_per_second'], 'probe': p['decode_kind']})
with (root / 'comparison.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def chart(phase):
    points = {name: [p for p in phases if p['phase'] == phase and p['tokens_per_second'] is not None]
              for name, phases in runs.items()}
    ymax = max(p['tokens_per_second'] for values in points.values() for p in values) * 1.12
    x = lambda c: 80 + c / 32768 * 780
    y = lambda t: 340 - t / ymax * 270
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 420" role="img">',
             '<rect width="920" height="420" fill="white"/>',
             f'<text x="80" y="30" font-size="22" font-family="sans-serif">{phase.capitalize()} throughput vs. actual context</text>']
    for i in range(6):
        value = ymax * i / 5
        parts += [f'<path d="M80 {y(value):.2f}H860" stroke="#e2e8f0"/>',
                  f'<text x="69" y="{y(value)+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{value:.1f}</text>']
    for value in (0,8192,16384,24576,32768):
        parts.append(f'<text x="{x(value)}" y="365" text-anchor="middle" font-family="sans-serif" font-size="13">{value:,}</text>')
    if divergence_boundary is not None:
        boundary=x(divergence_boundary)
        parts.append(f'<path d="M{boundary:.2f} 70V340" stroke="#64748b" stroke-dasharray="5 5"/>')
        parts.append(f'<text x="{boundary+8:.2f}" y="88" fill="#475569" font-size="12" font-family="sans-serif">Later prompts differ</text>')
    parts += ['<text x="470" y="399" text-anchor="middle" font-family="sans-serif">Total context tokens (input + generated)</text>',
              '<text x="20" y="210" transform="rotate(-90 20 210)" text-anchor="middle" font-family="sans-serif">Tokens / second</text>']
    for idx, (name, values) in enumerate(points.items()):
        line = ' '.join(f"{x(p['context_end_tokens']):.2f},{y(p['tokens_per_second']):.2f}" for p in values)
        parts.append(f'<polyline points="{line}" fill="none" stroke="{colors[name]}" stroke-width="2.5"/>')
        for p in values:
            rate = p['tokens_per_second']
            tip = f"{name}: {p['context_end_tokens']:,} tokens; {rate:.3f} tok/s; {p['tokens']} measured tokens"
            r = 6 if p['decode_kind'] == 'long' else 4
            parts.append(f'<circle cx="{x(p["context_end_tokens"]):.2f}" cy="{y(rate):.2f}" r="{r}" fill="{colors[name]}"><title>{html.escape(tip)}</title></circle>')
        lx=540+idx*155
        parts.append(f'<text x="{lx}" y="53" fill="{colors[name]}" font-family="sans-serif" font-size="15">● {name}</text>')
    parts.append('</svg>')
    svg='\n'.join(parts)
    (root / f'{phase}-comparison.svg').write_text(svg, encoding='utf-8')
    return svg


summary=[]
native={name:[json.loads(line) for line in (root/name/'native-timings.jsonl').read_text().splitlines()]
        for name in runs}
if len(native['pinned']) != len(native['pageable']):
    raise RuntimeError('Different probe counts; comparison needs investigation')
prompt_matches=[a['prompt_sha256']==b['prompt_sha256'] for a,b in zip(native['pinned'],native['pageable'])]
decode_matches=[a['generated_sha256']==b['generated_sha256'] for a,b in zip(native['pinned'],native['pageable'])]
verification={'probes':len(prompt_matches),'identical_prompts':sum(prompt_matches),
              'identical_generated_sequences':sum(decode_matches),
              'prompt_matches_by_cycle':prompt_matches,'generation_matches_by_cycle':decode_matches}
decode_phases=[p for p in runs['pinned'] if p['phase']=='decode']
divergence_boundary=next((p['context_end_tokens'] for p,match in zip(decode_phases,decode_matches) if not match),None)
svgs = [chart(phase) for phase in ('prefill','decode')]
(root/'workload-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
checkpoints={name:json.loads((root/name/'checkpoint.json').read_text())['tokens'] for name in runs}
token_rows={name:list(csv.DictReader((root/name/'tokens.csv').open())) for name in runs}
matched_summary=[]
for name,phases in runs.items():
    prefill=[p for p,match in zip([p for p in phases if p['phase']=='prefill'],prompt_matches) if match]
    matched_tokens=0
    matched_duration=0.0
    prefix_counts=[]
    for index,p in enumerate([p for p in phases if p['phase']=='decode']):
        if not prompt_matches[index]:
            prefix_counts.append(0)
            continue
        start=p['context_start_tokens']
        end=p['context_end_tokens']
        same_prefix=0
        for a,b in zip(checkpoints['pinned'][start:end],checkpoints['pageable'][start:end]):
            if a!=b:
                break
            same_prefix+=1
        prefix_counts.append(same_prefix)
        events=[t for t in token_rows[name] if int(t['phase_id'])==p['phase_id'] and int(t['token_index'])<=same_prefix]
        if len(events)>1:
            matched_tokens+=len(events)-1
            matched_duration+=float(events[-1]['elapsed_s'])-float(events[0]['elapsed_s'])
    matched_summary.append({'variant':name,'identical_prompt_probes':len(prefill),
        'identical_prompt_prefill_tps':sum(p['tokens']+p['cache_catchup_tokens'] for p in prefill)/sum(p['duration_s'] for p in prefill),
        'identical_decode_prefix_counts':prefix_counts,'identical_decode_interval_tokens':matched_tokens,
        'identical_decode_prefix_tps':matched_tokens/matched_duration})
(root/'matched-workload-summary.json').write_text(json.dumps(matched_summary,indent=2)+'\n')
for name, phases in runs.items():
    telemetry=[json.loads(line) for line in (root/name/'windows-telemetry.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    item={'variant':name}
    for phase in ('prefill','decode'):
        records=[p for p in phases if p['phase']==phase]
        numerator=sum(p['tokens'] + p['cache_catchup_tokens'] if phase=='prefill' else max(0,p['tokens']-1) for p in records)
        item[phase+'_weighted_tps']=numerator/sum(p['duration_s'] for p in records)
        later=records[1:]
        later_numerator=sum(p['tokens'] + p['cache_catchup_tokens'] if phase=='prefill' else max(0,p['tokens']-1) for p in later)
        item[phase+'_excluding_first_tps']=later_numerator/sum(p['duration_s'] for p in later)
    for key in ('dedicated_bytes','shared_bytes','working_set_bytes','free_ram_bytes'):
        item[key+'_median_gib']=statistics.median(r[key] for r in telemetry)/2**30
    summary.append(item)
(root/'comparison-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
body=''.join(svgs)
table='<table><tr><th>Variant</th><th>Prefill tok/s</th><th>Decode tok/s</th><th>Dedicated GiB</th><th>Shared GiB</th></tr>'
for s in summary:
    table+=f'<tr><td>{s["variant"]}</td><td>{s["prefill_weighted_tps"]:.2f}</td><td>{s["decode_weighted_tps"]:.2f}</td><td>{s["dedicated_bytes_median_gib"]:.2f}</td><td>{s["shared_bytes_median_gib"]:.3f}</td></tr>'
table+='</table>'
table+='<p>Excluding the first probe: '+ '; '.join(
    f'{s["variant"]}: {s["prefill_excluding_first_tps"]:.2f} prefill / {s["decode_excluding_first_tps"]:.2f} decode tok/s'
    for s in summary)+'. This reduces, but does not eliminate, the warm-server versus restart confound.</p>'
table+=f'<p>Workload verification: {sum(prompt_matches)}/{len(prompt_matches)} prompt sequences and {sum(decode_matches)}/{len(decode_matches)} generated sequences match exactly by SHA-256.</p>'
if divergence_boundary is not None:
    table+='<p><strong>Continuation caveat:</strong> Generated outputs diverged during a decode probe. Because the normal harness feeds generated tokens into later context, subsequent prompt contents differ. The dashed line marks the end of the first divergent decode probe. Later points compare end-to-end growing-context runs, not identical token workloads; do not attribute their differences solely to pinned memory.</p>'
    table+='<p>Identical-workload subset: '+ '; '.join(f'{s["variant"]}: {s["identical_prompt_prefill_tps"]:.2f} prefill / {s["identical_decode_prefix_tps"]:.2f} decode tok/s' for s in matched_summary)+'. Prefill uses matching prompt probes; decode uses only their identical generated prefixes (see matched-workload-summary.json).</p>'
document='''<!doctype html><meta charset="utf-8"><title>Distributed Qwen: pinned vs pageable host memory</title>
<style>body{font:16px system-ui;max-width:1000px;margin:36px auto;padding:0 20px;background:#f1f5f9;color:#172033}svg{width:100%;margin:15px 0;border-radius:12px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}p{line-height:1.6}</style>
<h1>Distributed Qwen: pinned vs pageable host memory</h1>
<p>Same model, two machines, split 5,26,4, 35 offload layers, 100K server capacity, q8_0 KV, batch 512 / micro-batch 128. Sweep stops at 32,768 actual tokens. Only the local CUDA pinned-host allocation setting changes. Worker settings remain constant.</p>
<p>Repository continuous-cache harness: 4,096-token appends, 128-token decode probes, 512-token probes at start, midpoint and endpoint. Prefill uses native server timing; decode uses exact streamed token intervals. Larger dots mark sustained probes. Hover dots for values.</p>'''+table+body+'''<p>One sweep per variant, pinned first on an already-warm server, pageable second after a restart. The first point is particularly sensitive to initialization. Rates are observations, not confidence intervals. Aggregate rates use total processed tokens divided by summed phase time. Initial prompt cache is discarded. Prompt and generated-token hashes are retained in each native-timings.jsonl for cross-run verification. Windows telemetry is separate from the harness's WSL-client-only memory figures; generic swap-limit fields in the individual harness reports do not establish Windows-server paging behavior.</p>'''
(root/'comparison.html').write_text(document, encoding='utf-8')
print(json.dumps(summary,indent=2))
