"""Blinded explanation-rating pack for two independent raters.

Pools evidence items (feature + grid_cells + description + the model's stated
label) from the cte_p1 family of runs -- gemma cte_p1, the gemma stability
replicates, and the Track B gpt-4o run -- samples 100, strips tile ID / true
label / run / condition, shuffles with a fixed seed, and writes:

  rating/rater_A.csv, rating/rater_B.csv   identical blank scoring sheets
  rating/images/r###.png                   gridded tile per item, anonymised name
  rating/rating_form.html                  self-contained local form (loads the
                                           images by relative path, autosaves to
                                           localStorage, exports CSV)
  rating/KEY_do_not_open_until_scored.csv  rating_id -> provenance + true label

Axes and value sets come from rubric.yaml `scoring.axes`. Raters score the
EXPLANATION against the rubric, not the tile against the truth.

    python3 make_rating_sheet.py
"""

import csv
import html
import json
import os
import random
import shutil

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(ROOT, "rating")
SEED = 20260806
N = 100

RUNS = [
    ("gemma_cte_p1", "runs/cte_p1.jsonl"),
    ("gemma_cte_p1_stability", "runs/cte_p1_stability.jsonl"),
    ("gpt4o_trackB_cte_p1", "runs/cte_p1_trackB.jsonl"),
]


def pool_items():
    items = []
    for run_id, rel in RUNS:
        for line in open(os.path.join(ROOT, rel)):
            r = json.loads(line)
            p = r.get("parsed")
            if not p or p.get("label") not in ("HP", "SSA"):
                continue
            for k, e in enumerate(p.get("evidence") or []):
                cells = e.get("grid_cells_valid") or []
                if not e.get("feature") or not cells or not (e.get("description") or "").strip():
                    continue
                items.append(
                    {
                        "run": run_id,
                        "image": r["image"],
                        "replicate": r["replicate"],
                        "evidence_index": k,
                        "true_label": r["label_true"],
                        "feature": e["feature"],
                        "grid_cells": " ".join(cells),
                        "description": e["description"].strip(),
                        "stated_label": p["label"],
                    }
                )
    return items


def main():
    rng = random.Random(SEED)
    items = pool_items()
    print(f"pooled {len(items)} evidence items from {len(RUNS)} runs")
    sample = rng.sample(items, N)
    rng.shuffle(sample)

    os.makedirs(os.path.join(OUTDIR, "images"), exist_ok=True)
    rubric = yaml.safe_load(open(os.path.join(ROOT, "rubric.yaml")))
    axes = rubric["scoring"]["axes"]

    rows = []
    for i, it in enumerate(sample, 1):
        rid = f"r{i:03d}"
        shutil.copyfile(
            os.path.join(ROOT, "gridded", it["image"].replace(".png", "_grid.png")),
            os.path.join(OUTDIR, "images", f"{rid}.png"),
        )
        rows.append({"rating_id": rid, **it})

    sheet_cols = [
        "rating_id", "image_file", "feature", "grid_cells", "description",
        "stated_label", "presence", "relevance", "support", "notes",
    ]
    for rater in ("A", "B"):
        with open(os.path.join(OUTDIR, f"rater_{rater}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sheet_cols)
            w.writeheader()
            for r in rows:
                w.writerow(
                    {
                        "rating_id": r["rating_id"],
                        "image_file": f"images/{r['rating_id']}.png",
                        "feature": r["feature"],
                        "grid_cells": r["grid_cells"],
                        "description": r["description"],
                        "stated_label": r["stated_label"],
                        "presence": "",
                        "relevance": "",
                        "support": "",
                        "notes": "",
                    }
                )

    key_cols = ["rating_id", "run", "image", "replicate", "evidence_index", "true_label"]
    with open(os.path.join(OUTDIR, "KEY_do_not_open_until_scored.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=key_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    write_form(rows, axes)
    print(f"wrote {len(rows)} items -> {OUTDIR}/")


def write_form(rows, axes):
    ax_defs = [
        ("presence", axes["presence"]["question"], axes["presence"]["values"]),
        ("relevance", axes["relevance"]["question"], axes["relevance"]["values"]),
        ("support", axes["support"]["question"], axes["support"]["values"]),
    ]
    cards = []
    for r in rows:
        radios = ""
        for ax, q, vals in ax_defs:
            opts = "".join(
                f'<label><input type="radio" name="{r["rating_id"]}_{ax}" '
                f'value="{v}"> {v}</label> '
                for v in map(str, vals)
            )
            radios += f'<div class="axis"><span class="q">{html.escape(q)}</span><br>{opts}</div>'
        cards.append(f"""
<div class="card" id="{r['rating_id']}">
  <div class="imgcol"><img src="images/{r['rating_id']}.png" width="336"></div>
  <div class="txtcol">
    <div class="rid">{r['rating_id']}</div>
    <div class="ev"><b>{html.escape(r['feature'])}</b> in cells
      <b>{html.escape(r['grid_cells'])}</b></div>
    <div class="desc">&ldquo;{html.escape(r['description'])}&rdquo;</div>
    <div class="lab">Model's stated label: <b>{r['stated_label']}</b></div>
    {radios}
    <input class="notes" placeholder="notes (optional)" name="{r['rating_id']}_notes">
  </div>
</div>""")

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Blinded explanation rating</title>
<style>
 body {{ font: 14px/1.45 -apple-system, sans-serif; margin: 24px; background:#fafafa; }}
 .card {{ display:flex; gap:20px; background:#fff; border:1px solid #ddd;
          border-radius:8px; padding:16px; margin-bottom:18px; }}
 .rid {{ color:#888; font-size:12px; }}
 .desc {{ margin:6px 0; font-style:italic; }}
 .axis {{ margin:8px 0; }} .q {{ font-weight:600; font-size:13px; }}
 .notes {{ width:95%; margin-top:6px; }}
 #bar {{ position:sticky; top:0; background:#fff; border-bottom:2px solid #ccc;
        padding:10px 0; margin-bottom:16px; z-index:9; }}
 button {{ font-size:14px; padding:6px 14px; }}
</style></head><body>
<div id="bar">
 <b>Blinded explanation rating</b> &nbsp; Rater initials:
 <input id="rater" size="6"> &nbsp; <span id="prog"></span> &nbsp;
 <button onclick="exportCsv()">Export CSV</button>
 <div style="font-size:12px;color:#555;margin-top:4px">
  Score each explanation against rubric.yaml. Answers autosave locally.
  presence: is the cited feature visible in the cited cells? &middot;
  relevance: per the rubric, is it diagnostically relevant AS DESCRIBED? &middot;
  support: does it support the model's stated label?
 </div>
</div>
{''.join(cards)}
<script>
const ids = {json.dumps([r["rating_id"] for r in rows])};
const axes = ["presence","relevance","support"];
document.addEventListener("change", save);
document.addEventListener("input", save);
function save() {{
  const s = {{rater: document.getElementById("rater").value}};
  for (const id of ids) {{
    for (const ax of axes) {{
      const el = document.querySelector(`input[name="${{id}}_${{ax}}"]:checked`);
      if (el) s[`${{id}}_${{ax}}`] = el.value;
    }}
    const n = document.querySelector(`input[name="${{id}}_notes"]`);
    if (n && n.value) s[`${{id}}_notes`] = n.value;
  }}
  localStorage.setItem("rating_state", JSON.stringify(s));
  const done = ids.filter(id => axes.every(ax => s[`${{id}}_${{ax}}`])).length;
  document.getElementById("prog").textContent = `${{done}}/${{ids.length}} complete`;
}}
(function restore() {{
  const s = JSON.parse(localStorage.getItem("rating_state") || "{{}}");
  if (s.rater) document.getElementById("rater").value = s.rater;
  for (const [k,v] of Object.entries(s)) {{
    if (k === "rater") continue;
    if (k.endsWith("_notes")) {{
      const n = document.querySelector(`input[name="${{k}}"]`); if (n) n.value = v;
    }} else {{
      const el = document.querySelector(`input[name="${{k}}"][value="${{v}}"]`);
      if (el) el.checked = true;
    }}
  }}
  save();
}})();
function exportCsv() {{
  const s = JSON.parse(localStorage.getItem("rating_state") || "{{}}");
  let out = "rating_id,rater,presence,relevance,support,notes\\n";
  for (const id of ids) {{
    const row = [id, s.rater||"", ...axes.map(ax => s[`${{id}}_${{ax}}`]||""),
                 (s[`${{id}}_notes`]||"").replace(/"/g,'""')];
    out += row.map(v => `"${{v}}"`).join(",") + "\\n";
  }}
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([out], {{type:"text/csv"}}));
  a.download = `ratings_${{s.rater||"anon"}}.csv`;
  a.click();
}}
</script></body></html>"""
    with open(os.path.join(OUTDIR, "rating_form.html"), "w") as fh:
        fh.write(doc)


if __name__ == "__main__":
    main()
