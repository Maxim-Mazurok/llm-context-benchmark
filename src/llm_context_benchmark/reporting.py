from __future__ import annotations

import csv
import html
import json
import math
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import MemorySample, PhaseResult

GIB = 1024**3


def write_records(path: Path, records: list[Any]) -> None:
    dictionaries = [record.to_dict() for record in records]
    path.with_suffix(".json").write_text(json.dumps(dictionaries, indent=2) + "\n")
    if not records:
        path.with_suffix(".csv").write_text("")
        return
    names = [field.name for field in fields(records[0])]
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(dictionaries)


def analyze(
    phases: list[PhaseResult],
    *,
    initial_swap: int | None,
    practical_decode_ratio: float,
    practical_swap_growth_bytes: int,
    no_swap_epsilon_bytes: int,
    stop_reason: str,
    hard_limit: int | None,
) -> dict[str, Any]:
    completed = [p for p in phases if p.status == "completed"]
    decodes = [p for p in completed if p.phase == "decode" and p.tokens_per_second]
    baseline_phase = decodes[0] if decodes else None
    baseline = baseline_phase.tokens_per_second if baseline_phase else None

    max_tested = max((p.context_end_tokens for p in completed), default=0)
    no_swap_limit = 0
    under_1g_limit = 0
    practical = 0
    initial_swap_value = initial_swap or 0
    threshold_contexts: dict[str, int | None] = {
        "90_percent": None,
        "80_percent": None,
        "50_percent": None,
    }

    for phase in decodes:
        growth = max(
            0,
            phase.swap_growth_peak_bytes
            if phase.swap_growth_peak_bytes is not None
            else (phase.swap_used_end_bytes or initial_swap_value)
            - initial_swap_value,
        )
        if growth <= no_swap_epsilon_bytes:
            no_swap_limit = max(no_swap_limit, phase.context_end_tokens)
        if growth <= GIB:
            under_1g_limit = max(under_1g_limit, phase.context_end_tokens)
        if baseline:
            ratio = phase.tokens_per_second / baseline
            for label, value in (
                ("90_percent", 0.9),
                ("80_percent", 0.8),
                ("50_percent", 0.5),
            ):
                if ratio < value and threshold_contexts[label] is None:
                    threshold_contexts[label] = phase.context_end_tokens
            if (
                ratio >= practical_decode_ratio
                and growth <= practical_swap_growth_bytes
            ):
                practical = max(practical, phase.context_end_tokens)

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "max_tested_context_tokens": max_tested,
        "hard_model_runtime_limit_tokens": hard_limit,
        "stop_reason": stop_reason,
        "baseline_decode_tokens_per_second": baseline,
        "baseline_decode_context_tokens": (
            baseline_phase.context_end_tokens if baseline_phase else None
        ),
        "baseline_decode_kind": baseline_phase.decode_kind if baseline_phase else None,
        "no_swap_limit_tokens": no_swap_limit or None,
        "under_1_gib_swap_limit_tokens": under_1g_limit or None,
        "decode_degradation_first_below": threshold_contexts,
        "practical_context_tokens": practical or None,
        "practical_definition": {
            "decode_at_least_baseline_ratio": practical_decode_ratio,
            "swap_growth_at_most_bytes": practical_swap_growth_bytes,
            "allocation_failure": False,
        },
        "initial_swap_used_bytes": initial_swap,
        "peak_swap_growth_bytes": max(
            (max(0, p.swap_growth_peak_bytes or 0) for p in completed), default=0
        ),
        "peak_swap_used_bytes": max(
            (p.swap_used_peak_bytes or 0 for p in completed), default=0
        )
        or None,
        "swapout_bytes": sum(
            max(0, p.swapout_delta_bytes or 0) for p in completed
        ),
        "worst_memory_pressure": max(
            (p.pressure_worst for p in completed),
            key=lambda value: {None: -1, "normal": 0, "warning": 1, "critical": 2}.get(value, -1),
            default=None,
        ),
    }


def _fmt_bytes(value: float | None) -> str:
    return "n/a" if value is None else f"{value / GIB:.2f} GiB"


def _fmt_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.1f} tok/s"


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    metadata: dict[str, Any],
    phases: list[PhaseResult],
) -> None:
    lines = [
        "# LLM context benchmark report",
        "",
        f"- Model: `{metadata.get('model')}`",
        f"- Adapter: `{metadata.get('adapter')}`",
        f"- Max tested context: **{summary['max_tested_context_tokens']:,} tokens**",
        f"- Practical context: **{summary.get('practical_context_tokens') or 'not established'}**",
        f"- Runtime/model hard limit: **{summary.get('hard_model_runtime_limit_tokens') or 'unknown'}**",
        f"- Stop reason: `{summary['stop_reason']}`",
        f"- Baseline decode: **{_fmt_rate(summary.get('baseline_decode_tokens_per_second'))}**",
        f"- No-swap limit: **{summary.get('no_swap_limit_tokens') or 'not established'}**",
        f"- <1 GiB swap-growth limit: **{summary.get('under_1_gib_swap_limit_tokens') or 'not established'}**",
        "",
        "## Charts",
        "",
        "[Open the interactive benchmark dashboard](dashboard.html)",
        "",
        "![Prefill speed versus context](prefill-speed-vs-context.svg)",
        "",
        "![Decode speed versus context](decode-speed-vs-context.svg)",
        "",
        "![Memory indicators versus context](memory-indicators-vs-context.svg)",
        "",
        "## Completed phases",
        "",
        "| Context | Phase | Kind | Tokens | Duration | Throughput | RSS max | Compressed delta | Swap delta |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in phases:
        lines.append(
            f"| {p.context_end_tokens:,} | {p.phase} | {p.decode_kind or '—'} | {p.tokens:,} | "
            f"{p.duration_s:.3f}s | {_fmt_rate(p.tokens_per_second)} | {_fmt_bytes(p.rss_max_bytes)} | "
            f"{_fmt_bytes(p.compressed_delta_bytes)} | {_fmt_bytes(p.swap_delta_bytes)} |"
        )
    lines += [
        "",
        "## Files",
        "",
        (
            "`samples.csv` and `samples.jsonl` contain the raw memory time series. "
            "`phases.csv`/`phases.json` contain phase aggregates, while "
            "`tokens.csv`/`tokens.json` contain per-token decode timing. "
            "`dashboard.html` contains the interactive charts. "
            "`summary.json` and `run-metadata.json` preserve the derived result and configuration."
        ),
        "",
        (
            "The practical context is mechanically selected from completed decode phases using the thresholds in `summary.json`. "
            "Raw data is retained so those criteria can be changed later without rerunning inference."
        ),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def _svg_chart(
    path: Path,
    title: str,
    series: list[tuple[str, str, list[tuple[float, float]]]],
    y_label: str,
) -> None:
    width, height = 1000, 520
    left, right, top, bottom = 90, 30, 55, 70
    points = [point for _, _, values in series for point in values]
    if not points:
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="40" y="60">No data</text></svg>'
        )
        return
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(0.0, min(ys)), max(ys)
    if x_max == x_min:
        x_max += 1
    if y_max == y_min:
        y_max += 1
    plot_w, plot_h = width - left - right, height - top - bottom
    sx = lambda x: left + (x - x_min) / (x_max - x_min) * plot_w
    sy = lambda y: top + plot_h - (y - y_min) / (y_max - y_min) * plot_h
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#111827"/>',
        f'<text x="{left}" y="32" fill="#f9fafb" font-family="system-ui" font-size="22" font-weight="600">{html.escape(title)}</text>',
    ]
    for i in range(6):
        x = left + plot_w * i / 5
        value = x_min + (x_max - x_min) * i / 5
        out += [
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#374151"/>',
            f'<text x="{x:.1f}" y="{top + plot_h + 28}" fill="#9ca3af" font-family="system-ui" font-size="12" text-anchor="middle">{value / 1000:.0f}K</text>',
        ]
    for i in range(6):
        y = top + plot_h * i / 5
        value = y_max - (y_max - y_min) * i / 5
        out += [
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#374151"/>',
            f'<text x="{left - 12}" y="{y + 4:.1f}" fill="#9ca3af" font-family="system-ui" font-size="12" text-anchor="end">{value:.1f}</text>',
        ]
    for name, color, values in series:
        if values:
            coords = " ".join(
                f"{sx(x):.1f},{sy(y):.1f}" for x, y in values if math.isfinite(y)
            )
            out.append(
                f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>'
            )
    legend_x = left
    for name, color, _ in series:
        out += [
            f'<rect x="{legend_x}" y="{height - 25}" width="14" height="4" fill="{color}"/>',
            f'<text x="{legend_x + 20}" y="{height - 18}" fill="#d1d5db" font-family="system-ui" font-size="12">{html.escape(name)}</text>',
        ]
        legend_x += 145
    out += [
        f'<text x="{left + plot_w / 2}" y="{height - 42}" fill="#d1d5db" font-family="system-ui" font-size="13" text-anchor="middle">Actual total context (tokens)</text>',
        f'<text x="20" y="{top + plot_h / 2}" fill="#d1d5db" font-family="system-ui" font-size="13" text-anchor="middle" transform="rotate(-90 20 {top + plot_h / 2})">{html.escape(y_label)}</text>',
        "</svg>",
    ]
    path.write_text("\n".join(out))


def _sample_value(sample: MemorySample | dict[str, Any], name: str) -> Any:
    return sample.get(name) if isinstance(sample, dict) else getattr(sample, name)


def _memory_series(
    phases: list[PhaseResult],
    samples: list[MemorySample | dict[str, Any]],
    initial_swap: int | None,
) -> list[tuple[str, str, list[tuple[float, float]]]]:
    last_samples: dict[tuple[int, str], MemorySample | dict[str, Any]] = {}
    for sample in samples:
        phase_id = int(_sample_value(sample, "phase_id"))
        phase = str(_sample_value(sample, "phase"))
        last_samples[(phase_id, phase)] = sample
    decodes = [p for p in phases if p.phase == "decode" and p.status == "completed"]

    def phase_points(attribute: str) -> list[tuple[float, float]]:
        return [
            (p.context_end_tokens, value / GIB)
            for p in decodes
            if (value := getattr(p, attribute)) is not None
        ]

    available: list[tuple[float, float]] = []
    for phase in decodes:
        sample = last_samples.get((phase.phase_id, "decode"))
        if sample is None:
            continue
        value = _sample_value(sample, "system_available_bytes")
        if value not in (None, ""):
            available.append((phase.context_end_tokens, float(value) / GIB))
    swap_baseline = initial_swap or 0
    swap_growth = [
        (
            p.context_end_tokens,
            max(
                0,
                p.swap_growth_peak_bytes
                if p.swap_growth_peak_bytes is not None
                else (p.swap_used_end_bytes or swap_baseline) - swap_baseline,
            )
            / GIB,
        )
        for p in decodes
        if p.swap_used_end_bytes is not None
    ]
    return [
        ("MLX active max", "#b74228", phase_points("mlx_active_max_bytes")),
        ("System available", "#2f766f", available),
        ("Compressed", "#b67b00", phase_points("compressed_end_bytes")),
        ("Swap growth", "#8f3f65", swap_growth),
    ]


def _throughput_series(
    phases: list[PhaseResult],
) -> tuple[
    list[tuple[str, str, list[tuple[float, float]]]],
    list[tuple[str, str, list[tuple[float, float]]]],
]:
    prefill = [p for p in phases if p.phase == "prefill" and p.tokens_per_second]
    short = [
        p
        for p in phases
        if p.phase == "decode" and p.decode_kind == "short" and p.tokens_per_second
    ]
    long = [
        p
        for p in phases
        if p.phase == "decode" and p.decode_kind == "long" and p.tokens_per_second
    ]
    return (
        [
            (
                "Prefill",
                "#c24e2b",
                [(p.context_end_tokens, p.tokens_per_second) for p in prefill],
            )
        ],
        [
            (
                "Short decode",
                "#254b76",
                [(p.context_end_tokens, p.tokens_per_second) for p in short],
            ),
            (
                "Long decode",
                "#9b6b00",
                [(p.context_end_tokens, p.tokens_per_second) for p in long],
            ),
        ],
    )


def write_charts(
    output_dir: Path,
    phases: list[PhaseResult],
    samples: list[MemorySample | dict[str, Any]] | None = None,
    initial_swap: int | None = None,
) -> None:
    samples = samples or []
    memory_series = _memory_series(phases, samples, initial_swap)
    _svg_chart(
        output_dir / "memory-vs-context.svg",
        "Useful memory indicators versus actual context",
        memory_series,
        "GiB",
    )
    prefill_series, decode_series = _throughput_series(phases)
    _svg_chart(
        output_dir / "throughput-vs-context.svg",
        "Throughput versus actual context",
        prefill_series + decode_series,
        "tokens / second",
    )
    _svg_chart(
        output_dir / "prefill-speed-vs-context.svg",
        "Prefill speed versus actual context",
        prefill_series,
        "tokens / second",
    )
    _svg_chart(
        output_dir / "decode-speed-vs-context.svg",
        "Decode speed versus actual context",
        decode_series,
        "tokens / second",
    )
    _svg_chart(
        output_dir / "memory-indicators-vs-context.svg",
        "Useful memory indicators versus actual context",
        memory_series,
        "GiB",
    )


def write_dashboard(
    output_dir: Path,
    phases: list[PhaseResult],
    samples: list[MemorySample | dict[str, Any]],
    summary: dict[str, Any],
    metadata: dict[str, Any],
    initial_swap: int | None,
) -> None:
    prefill_series, decode_series = _throughput_series(phases)
    memory_series = _memory_series(phases, samples, initial_swap)

    def serialise(
        series: list[tuple[str, str, list[tuple[float, float]]]],
    ) -> list[dict[str, Any]]:
        return [
            {"name": name, "color": color, "points": points}
            for name, color, points in series
        ]

    data = json.dumps(
        {
            "prefill": serialise(prefill_series),
            "decode": serialise(decode_series),
            "memory": serialise(memory_series),
        },
        separators=(",", ":"),
    )
    model = html.escape(str(metadata.get("model", "unknown")))
    run_name = html.escape(output_dir.name)
    max_context = int(summary.get("max_tested_context_tokens") or 0)
    stop_reason = html.escape(str(summary.get("stop_reason", "unknown")))
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM context benchmark dashboard</title>
<style>
:root{color-scheme:light;--paper:oklch(96% .015 75);--sheet:oklch(99% .008 75);--ink:oklch(25% .025 55);--muted:oklch(49% .025 60);--rule:oklch(84% .02 70)}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:0.9rem/1.5 "Avenir Next",Avenir,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}
main{max-width:1280px;margin:auto;padding:clamp(20px,4vw,56px)}.hero{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:48px;align-items:end;margin-bottom:48px;border-bottom:2px solid var(--ink);padding-bottom:24px}
h1{font:600 clamp(2rem,5vw,4.25rem)/.95 Georgia,"Times New Roman",serif;letter-spacing:-.045em;margin:0 0 12px}.sub{color:var(--muted);word-break:break-all;max-width:70ch}.hero-side{display:grid;justify-items:end;gap:18px}.stats{display:flex;gap:28px}.pill{min-width:130px;border-top:1px solid var(--ink);padding-top:8px;color:var(--muted)}.pill b{display:block;color:var(--ink);font-size:1.15rem;margin-top:3px}
.grid{display:grid;grid-template-columns:1fr 1fr;column-gap:40px;row-gap:56px}.card{min-width:0;border-top:1px solid var(--rule);padding-top:16px;container-type:inline-size}.wide{grid-column:1/-1}h2{font:600 1.35rem/1.2 Georgia,"Times New Roman",serif;margin:0 0 5px}.hint{color:var(--muted);font-size:.78rem;letter-spacing:.02em;margin-bottom:12px}.chart{width:100%;min-height:360px;background:var(--sheet);border:1px solid var(--rule)}.legend{display:flex;gap:18px;flex-wrap:wrap;padding:0 16px 14px;color:var(--muted)}.legend span::before{content:"";display:inline-block;width:16px;height:3px;background:var(--c);vertical-align:middle;margin-right:7px}.note{color:var(--muted);margin:40px 0 0;max-width:75ch;line-height:1.65}
.delete-run,.dialog-actions button{font:600 .82rem/1 "Avenir Next",Avenir,"Segoe UI",sans-serif;padding:10px 14px;border:1px solid currentColor;background:transparent;color:oklch(45% .16 28);cursor:pointer}.delete-run:hover,.dialog-danger:hover{background:oklch(93% .035 28)}.delete-run:disabled{color:var(--muted);cursor:not-allowed;opacity:.55}.delete-help{font-size:.72rem;color:var(--muted);max-width:32ch;text-align:right}.delete-help:empty{display:none}
dialog{width:min(460px,calc(100vw - 32px));border:1px solid var(--ink);background:var(--sheet);color:var(--ink);padding:28px}dialog::backdrop{background:oklch(25% .025 55 / .48)}dialog h2{font-size:1.6rem;margin-bottom:12px}dialog p{color:var(--muted);margin:0 0 24px}.dialog-actions{display:flex;justify-content:flex-end;gap:10px}.dialog-cancel{color:var(--ink)!important}.dialog-status{min-height:1.5em;color:oklch(42% .16 28)!important}
svg{display:block;width:100%;height:auto}svg text{font-family:"Avenir Next",Avenir,"Segoe UI",sans-serif}.point{cursor:crosshair}.point:hover{stroke:var(--sheet);stroke-width:3}
@media(max-width:850px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}.hero{display:block;margin-bottom:32px}.hero-side{justify-items:start;margin-top:24px}.stats{margin-top:0}.delete-help{text-align:left}.chart{min-height:280px}}
@media(max-width:520px){.stats{display:grid;grid-template-columns:1fr 1fr;gap:16px}main{padding:20px 12px}.grid{row-gap:36px}}
</style></head><body><main>
<div class="hero"><div><h1>LLM context benchmark</h1><div class="sub">__MODEL__</div></div><div class="hero-side"><div class="stats"><div class="pill"><span>Max tested</span><b>__MAX_CONTEXT__</b></div><div class="pill"><span>Stopped</span><b>__STOP_REASON__</b></div></div><button id="deleteRun" class="delete-run" type="button" disabled>Delete run</button><span id="deleteHelp" class="delete-help">Open through the local dashboard viewer to enable deletion.</span></div></div>
<div class="grid"><section class="card"><h2>Prefill speed</h2><div class="hint">Input processing as the live cache grows</div><div id="prefill" class="chart"></div></section>
<section class="card"><h2>Decode speed</h2><div class="hint">Short probes and sustained 1K-token probes</div><div id="decode" class="chart"></div></section>
<section class="card wide"><h2>Memory indicators</h2><div class="hint">MLX is process-specific; available, compressed, and swap are system-wide</div><div id="memory" class="chart"></div></section></div>
<p class="note">Context includes both appended input and generated benchmark tokens. Hover a point for its exact value. Swap is growth relative to the post-model-load baseline.</p>
</main><dialog id="deleteDialog" aria-labelledby="deleteTitle"><form method="dialog"><h2 id="deleteTitle">Move this run to Trash?</h2><p id="deleteDetail">Run <strong>__RUN_NAME__</strong> and all of its raw samples, charts, and reports will be moved to macOS Trash.</p><p id="deleteStatus" class="dialog-status" aria-live="polite"></p><div class="dialog-actions"><button class="dialog-cancel" value="cancel">Keep run</button><button id="confirmDelete" class="dialog-danger" type="button">Move to Trash</button></div></form></dialog><script>
const DATA=__DATA__;
const NS="http://www.w3.org/2000/svg";
function el(tag,attrs={},text=""){const n=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text)n.textContent=text;return n}
function chart(id,series,yLabel){const host=document.getElementById(id),W=720,H=390,m={l:70,r:22,t:20,b:58};const all=series.flatMap(s=>s.points);if(!all.length){host.textContent="No data";return}const xmax=Math.max(...all.map(p=>p[0])),ymax=Math.max(...all.map(p=>p[1]))*1.08||1;const sx=x=>m.l+x/xmax*(W-m.l-m.r),sy=y=>H-m.b-y/ymax*(H-m.t-m.b);const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,role:"img"});
for(let i=0;i<=5;i++){const x=m.l+(W-m.l-m.r)*i/5,v=xmax*i/5;svg.append(el("line",{x1:x,y1:m.t,x2:x,y2:H-m.b,stroke:"#d8d1c4"}));svg.append(el("text",{x,y:H-m.b+25,fill:"#6f6a60","font-size":11,"text-anchor":"middle"},`${Math.round(v/1000)}K`))}
for(let i=0;i<=5;i++){const y=m.t+(H-m.t-m.b)*i/5,v=ymax*(1-i/5);svg.append(el("line",{x1:m.l,y1:y,x2:W-m.r,y2:y,stroke:"#d8d1c4"}));svg.append(el("text",{x:m.l-10,y:y+4,fill:"#6f6a60","font-size":11,"text-anchor":"end"},v>=100?Math.round(v):v.toFixed(1)))}
for(const s of series){const d=s.points.map((p,i)=>`${i?"L":"M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(" ");svg.append(el("path",{d,fill:"none",stroke:s.color,"stroke-width":2.5,"stroke-linejoin":"round"}));for(const p of s.points){const c=el("circle",{cx:sx(p[0]),cy:sy(p[1]),r:3.5,fill:s.color,class:"point"});c.append(el("title",{},`${s.name}\n${p[0].toLocaleString()} context\n${p[1].toFixed(2)} ${yLabel}`));svg.append(c)}}
svg.append(el("text",{x:(m.l+W-m.r)/2,y:H-10,fill:"#6f6a60","font-size":12,"text-anchor":"middle"},"Actual total context (tokens)"));const yl=el("text",{x:14,y:H/2,fill:"#6f6a60","font-size":12,"text-anchor":"middle",transform:`rotate(-90 14 ${H/2})`},yLabel);svg.append(yl);host.append(svg);const legend=document.createElement("div");legend.className="legend";for(const s of series){const item=document.createElement("span");item.style.setProperty("--c",s.color);item.textContent=s.name;legend.append(item)}host.append(legend)}
chart("prefill",DATA.prefill,"tok/s");chart("decode",DATA.decode,"tok/s");chart("memory",DATA.memory,"GiB");
const deleteToken=new URLSearchParams(location.search).get("token"),deleteButton=document.getElementById("deleteRun"),deleteHelp=document.getElementById("deleteHelp"),deleteDialog=document.getElementById("deleteDialog"),confirmDelete=document.getElementById("confirmDelete"),deleteStatus=document.getElementById("deleteStatus");
if(location.protocol.startsWith("http")&&deleteToken){deleteButton.disabled=false;deleteHelp.textContent=""}
deleteButton.addEventListener("click",()=>deleteDialog.showModal());
confirmDelete.addEventListener("click",async()=>{confirmDelete.disabled=true;confirmDelete.textContent="Moving…";deleteStatus.textContent="";try{const response=await fetch(`/api/delete-run?token=${encodeURIComponent(deleteToken)}`,{method:"POST"}),result=await response.json();if(!response.ok||!result.ok)throw new Error(result.error||"Deletion failed");document.getElementById("deleteTitle").textContent="Run moved to Trash";document.getElementById("deleteDetail").textContent="You can recover it from macOS Trash until Trash is emptied.";deleteStatus.textContent=result.movedTo;confirmDelete.remove();deleteDialog.querySelector(".dialog-cancel").textContent="Close"}catch(error){deleteStatus.textContent=error.message;confirmDelete.disabled=false;confirmDelete.textContent="Move to Trash"}});
</script></body></html>"""
    document = (
        document.replace("__DATA__", data)
        .replace("__MODEL__", model)
        .replace("__MAX_CONTEXT__", f"{max_context:,} tokens")
        .replace("__STOP_REASON__", stop_reason)
        .replace("__RUN_NAME__", run_name)
    )
    (output_dir / "dashboard.html").write_text(document)
