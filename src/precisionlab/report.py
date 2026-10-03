"""Portable, escaped HTML report generated only from measured predictions."""

from html import escape
import base64
import json
import mimetypes
from pathlib import Path


def render(result, output, *, title="VLM precision regression", dataset_root=None):
    overall = result["overall"]
    task_rows = "".join(f'<tr><td>{escape(task)}</td><td>{s["samples"]}</td><td>{s["reference_em"]:.1%}</td>'
        f'<td>{s["candidate_em"]:.1%}</td><td>{s["delta_pp"]:+.2f} pp</td><td>{s["regressions"]}</td><td>{s["recoveries"]}</td>'
        f'<td>{s["delta_ci95_pp"][0]:+.2f} to {s["delta_ci95_pp"][1]:+.2f}</td></tr>' for task, s in result["tasks"].items())
    semantic_rows = "".join(f'<tr><td>{escape(task)}</td><td>{s["reference_em"]:.1%}</td><td>{s["candidate_em"]:.1%}</td>'
        f'<td>{s["delta_pp"]:+.2f} pp</td><td>{s["regressions"]}</td><td>{s["recoveries"]}</td>'
        f'<td>{s["delta_ci95_pp"][0]:+.2f} to {s["delta_ci95_pp"][1]:+.2f}</td></tr>' for task, s in result["semantic_tasks"].items())
    cards = []
    for c in result["failures"]:
        picture = ""
        if dataset_root:
            root = Path(dataset_root).resolve()
            path = (root / c["image"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Report image escapes dataset root")
            mime = mimetypes.guess_type(path.name)[0]
            if mime not in {"image/png", "image/jpeg", "image/webp"}:
                raise ValueError("Only raster images can be embedded in the report")
            encoded = base64.b64encode(path.read_bytes()).decode()
            picture = f'<img alt="Original task input" style="max-width:100%;height:auto" src="data:{mime};base64,{encoded}">'
        kind = "Content preserved; strict format/order failed" if c["candidate_content_correct"] else "Content differs from expected"
        cards.append(f'<details><summary>{escape(c["task"])} · {escape(c["id"])}</summary><p class="note">{kind}</p>'
            f'{picture}<p>{escape(c["prompt"])}</p><p>Expected <code>{escape(c["answer"])}</code></p><div class="pair"><div>Reference<pre>{escape(c["reference"])}</pre></div>'
            f'<div>Candidate<pre>{escape(c["candidate"])}</pre></div></div></details>')
    failures = "".join(cards)
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>
body{{margin:0;background:#f4f6fa;color:#16243a;font:16px/1.5 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:40px 24px}}
h1{{font-size:36px;line-height:1.2;margin:8px 0 18px}}.eyebrow{{color:#355fc3;font-size:13px;letter-spacing:2px}}.grid,.pair{{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}}
.grid{{grid-template-columns:repeat(4,1fr);margin:28px 0}}.card,section,details{{background:white;border:1px solid #dde3ee;border-radius:12px;padding:20px}}.card strong{{display:block;font-size:30px}}
section{{margin-bottom:22px;overflow:auto}}h2{{font-size:21px}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #e7eaf0;white-space:nowrap}}
th{{color:#57677d}}details{{margin:10px 0}}summary{{cursor:pointer}}pre{{white-space:pre-wrap;background:#f5f7fb;padding:12px;border-radius:6px}}.note{{color:#5d6a7d}}@media(max-width:700px){{.grid,.pair{{grid-template-columns:1fr 1fr}}h1{{font-size:28px}}}}
</style><main><div class="eyebrow">VLM PRECISION LAB / PAIRED EVIDENCE</div><h1>{escape(title)}</h1>
<p class="note">{escape(result["interpretation"])}</p><div class="grid">
<div class="card">Reference EM<strong>{overall["reference_em"]:.1%}</strong></div><div class="card">Candidate EM<strong>{overall["candidate_em"]:.1%}</strong></div>
<div class="card">Paired regressions<strong>{overall["regressions"]}</strong></div><div class="card">Independent sources<strong>{overall["independent_groups"]}</strong></div></div>
<section><h2>Task-specific changes</h2><table><thead><tr><th>Task</th><th>N</th><th>Reference</th><th>Candidate</th><th>Change</th><th>Lost</th><th>Recovered</th><th>95% paired cluster CI (pp)</th></tr></thead><tbody>{task_rows}</tbody></table>
<p class="note">Cluster bootstrap by source; {result["bootstrap"]["repetitions"]:,} repetitions. Small source counts do not establish equivalence. Gains and losses are shown separately.</p></section>
<section><h2>Separate content from formatting</h2><p>{escape(result["semantic_metric"])}</p><table><thead><tr><th>Task</th><th>Reference</th><th>Candidate</th><th>Change</th><th>Lost</th><th>Recovered</th><th>95% paired cluster CI (pp)</th></tr></thead><tbody>{semantic_rows}</tbody></table>
<p class="note">A reversed endpoint order can fail strict exact match while preserving the relationship. Do not call all strict-score loss a perception regression.</p></section>
<section><h2>Reference-correct → candidate-wrong exact-match cases</h2>{failures or '<p>No paired regressions in this sample.</p>'}</section>
<p class="note">Exact match preserves signs, punctuation and decimal precision. This report does not establish a mechanism, speedup or deployment readiness by itself.</p></main></html>'''
    Path(output).write_text(html)
    Path(output).with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
