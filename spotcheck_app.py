"""Local click-through page for the experiment 7 and 8 spot checks.

    python spotcheck_app.py          then open http://127.0.0.1:8813/

Each answer is written straight into the experiment's CSV (the `verdict` column, plus `note` where the CSV
has one), so closing the page loses nothing. The page hides each chip's automatic result (caught, missed,
detected) so it can't steer the answer.
"""
from __future__ import annotations

import csv
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("SPOTCHECK_PORT", "8813"))
LOCK = threading.Lock()

SETS = {
    "exp7": {
        "csv": ROOT / "exp7_change_tasking" / "r2_spotcheck" / "chips.csv",
        "img_dir": ROOT / "exp7_change_tasking" / "r2_spotcheck",
        "img_col": "file",
        "title": "Experiment 7: is the Capella change real?",
        "question": "Look at the two Capella images and the purple change patch. Did something on the ground "
                    "really change (an object moved, was built, removed, flooded, dug)? Ignore the Sentinel-1 panels.",
        "choices": [["real", "Real change"], ["artifact", "Not real (shift, edge, speckle, layover)"],
                    ["natural", "Natural only (canopy, moisture, wind)"], ["unsure", "Unsure"]],
        "info": ["land", "area_m2", "sign", "capella_before", "capella_after"],
    },
    "exp8": {
        "csv": ROOT / "exp8_feature_width" / "spotcheck.csv",
        "img_dir": ROOT / "exp8_feature_width" / "spotcheck",
        "img_col": "panel",
        "title": "Experiment 8: is the feature there, and does Sentinel-1 show it?",
        "question": "Is there a linear feature of about the stated width at the red + on Capella? "
                    "If so, can you see it on Sentinel-1?",
        "choices": [["both", "On Capella and visible on Sentinel-1"], ["capella_only", "On Capella, not on Sentinel-1"],
                    ["no_feature", "No clear feature on Capella"], ["unsure", "Unsure"]],
        "info": ["type", "cap_width_m", "cap_contrast_db", "country"],
    },
    "exp9": {
        "csv": ROOT / "exp9_cluster_tasking" / "spotcheck" / "looks.csv",
        "img_dir": ROOT / "exp9_cluster_tasking" / "spotcheck",
        "img_col": "file",
        "title": "Experiment 9: would this Capella look find real change?",
        "question": "Look at the two Capella images in this 500 m look. Did something on the ground really change "
                    "(an object moved, was built, removed, flooded, dug)? Ignore the Sentinel-1 panels.",
        "choices": [["real", "Real change in the look"], ["none", "No real change"],
                    ["natural", "Natural only (canopy, moisture, wind)"], ["unsure", "Unsure"]],
        "info": ["site_id", "pair"],
    },
}


def read_rows(s):
    with open(s["csv"], newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def write_rows(s, fields, rows):
    tmp = s["csv"].with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, s["csv"])


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>SAR Spot Check</title>
<style>
:root{--bg:#f3f5f7;--panel:#fff;--fg:#18212b;--muted:#5b6875;--line:#d5dbe1;--accent:#1f6fd1;--on:#7e3fc0}
@media (prefers-color-scheme:dark){:root{--bg:#0f151b;--panel:#161e26;--fg:#e3e9ef;--muted:#93a1af;--line:#2a3541;--accent:#6aa8ff;--on:#b58cf0;color-scheme:dark}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 "Segoe UI",system-ui,sans-serif}
.wrap{max-width:1300px;margin:0 auto;padding:16px;display:grid;gap:12px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
button{font:inherit;padding:8px 12px;border-radius:6px;border:1px solid var(--line);background:var(--panel);color:var(--fg);cursor:pointer}
button.sel{border-color:var(--on);outline:2px solid var(--on)}
button.tab.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;min-width:0}
img{max-width:100%;height:auto;background:#fff;border-radius:4px}
.muted{color:var(--muted)} .choices{display:flex;flex-wrap:wrap;gap:8px}
input[type=text]{font:inherit;padding:7px;width:100%;box-sizing:border-box;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--fg)}
kbd{border:1px solid var(--line);border-radius:3px;padding:0 4px;font-size:.8em}
</style></head><body><div class="wrap">
<div class="bar"><b>Spot check</b><button class="tab" data-s="exp7">Exp 7: change</button><button class="tab" data-s="exp8">Exp 8: width</button><button class="tab" data-s="exp9">Exp 9: tasking</button>
<span id="prog" class="muted"></span></div>
<div class="card"><h2 id="title" style="margin:0 0 4px;font-size:1.1rem"></h2><div id="q" class="muted"></div></div>
<div class="card"><div class="bar"><button id="prev">&larr; Prev <kbd>A</kbd></button><span id="pos"></span>
<button id="next">Next <kbd>D</kbd> &rarr;</button><button id="todo">Next unanswered <kbd>N</kbd></button><span id="info" class="muted"></span></div></div>
<div class="card"><img id="img" alt="spot-check panel"></div>
<div class="card"><div id="choices" class="choices"></div>
<div style="margin-top:8px"><input id="note" type="text" placeholder="Optional note (saved when you pick an answer or press Enter)"></div>
<div id="saved" class="muted" style="margin-top:6px"></div></div>
</div><script>
let S="exp7", data={}, i=0;
const $=id=>document.getElementById(id);
async function load(s){ if(!data[s]) data[s]=await (await fetch("/api/"+s)).json(); S=s;
  const r=data[s].rows; i=r.findIndex(x=>!x.verdict); if(i<0) i=0; render(); }
function render(){ const d=data[S], r=d.rows[i];
  document.querySelectorAll(".tab").forEach(b=>b.classList.toggle("on",b.dataset.s==S));
  $("title").textContent=d.title; $("q").textContent=d.question;
  $("img").src="/img/"+S+"/"+encodeURIComponent(r._img);
  $("pos").textContent=(i+1)+" / "+d.rows.length;
  $("info").textContent=d.info.map(k=>k+": "+(r[k]??"")).join("  ·  ");
  const done=d.rows.filter(x=>x.verdict).length; $("prog").textContent=done+" of "+d.rows.length+" answered";
  $("choices").innerHTML=""; d.choices.forEach(([v,l],k)=>{ const b=document.createElement("button");
    b.textContent=(k+1)+". "+l; if(r.verdict==v) b.classList.add("sel"); b.onclick=()=>save(v); $("choices").appendChild(b); });
  $("note").value=r.note||""; $("saved").textContent=r.verdict?"Saved: "+r.verdict:""; }
async function save(v){ const r=data[S].rows[i]; v=v??r.verdict; if(!v) return;
  const note=$("note").value; const res=await fetch("/api/"+S,{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({idx:i,verdict:v,note})});
  if(!res.ok){ $("saved").textContent="Save failed: "+res.status; return; }
  r.verdict=v; r.note=note; if(i<data[S].rows.length-1) i++; render(); }
$("prev").onclick=()=>{ if(i>0){i--;render();} }; $("next").onclick=()=>{ if(i<data[S].rows.length-1){i++;render();} };
$("todo").onclick=()=>{ const r=data[S].rows; const j=r.findIndex((x,k)=>k>i&&!x.verdict); const k=j>=0?j:r.findIndex(x=>!x.verdict); if(k>=0){i=k;render();} };
document.querySelectorAll(".tab").forEach(b=>b.onclick=()=>load(b.dataset.s));
document.addEventListener("keydown",e=>{ if(e.target.id=="note"){ if(e.key=="Enter") save(); return; }
  const n=parseInt(e.key); if(n>=1&&n<=data[S].choices.length) save(data[S].choices[n-1][0]);
  else if(e.key=="a"||e.key=="ArrowLeft") $("prev").click(); else if(e.key=="d"||e.key=="ArrowRight") $("next").click();
  else if(e.key=="n") $("todo").click(); });
load("exp7");
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = unquote(urlparse(self.path).path)
        if p == "/":
            return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
        parts = p.strip("/").split("/")
        if len(parts) == 2 and parts[0] == "api" and parts[1] in SETS:
            s = SETS[parts[1]]
            with LOCK:
                _, rows = read_rows(s)
            for r in rows:
                r["_img"] = r[s["img_col"]]
            out = {k: s[k] for k in ("title", "question", "choices", "info")} | {"rows": rows}
            return self.send(200, json.dumps(out).encode(), "application/json")
        if len(parts) == 3 and parts[0] == "img" and parts[1] in SETS:
            d = SETS[parts[1]]["img_dir"]
            f = (d / parts[2]).resolve()
            if f.parent == d.resolve() and f.suffix == ".png" and f.exists():
                return self.send(200, f.read_bytes(), "image/png")
        self.send(404, b"not found", "text/plain")

    def do_POST(self):
        parts = urlparse(self.path).path.strip("/").split("/")
        if len(parts) != 2 or parts[0] != "api" or parts[1] not in SETS:
            return self.send(404, b"not found", "text/plain")
        s = SETS[parts[1]]
        msg = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        valid = {c[0] for c in s["choices"]}
        if msg.get("verdict") not in valid:
            return self.send(400, b"bad verdict", "text/plain")
        with LOCK:
            fields, rows = read_rows(s)
            idx = int(msg["idx"])
            rows[idx]["verdict"] = msg["verdict"]
            if "note" in fields:
                rows[idx]["note"] = msg.get("note", "")
            elif msg.get("note"):
                fields = fields + ["note"]
                for r in rows:
                    r.setdefault("note", "")
                rows[idx]["note"] = msg["note"]
            write_rows(s, fields, rows)
        self.send(200, b"ok", "text/plain")


if __name__ == "__main__":
    print(f"Spot check: http://127.0.0.1:{PORT}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
