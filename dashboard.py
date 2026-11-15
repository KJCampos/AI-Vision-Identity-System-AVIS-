"""
dashboard.py - AVIS control panel in your browser (local only: http://127.0.0.1:8765).

People (with "Forget" = delete their face data), recent events + snapshots,
graphs/reports from runs/, and a settings editor for config.yaml.
Only reachable from this PC; changes need a per-session token.

    python dashboard.py
"""
import html
import json
import os
import secrets
import shutil
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from avis.learning import FaceDB  # noqa: E402

PORT = 8765
TOKEN = secrets.token_hex(16)
SERVE_DIRS = ("dataset", "runs")
FACES = os.path.join(HERE, "faces_db")
CONFIG = os.path.join(HERE, "config.yaml")


def people():
    db = FaceDB(FACES)
    out = []
    for name in sorted(db.people):
        d = os.path.join(HERE, "dataset", name)
        imgs, runs = [], []
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                jpgs = sorted(f for f in files if f.lower().endswith(".jpg"))
                if jpgs:
                    rel = os.path.relpath(root, os.path.join(HERE, "dataset")).replace(os.sep, "/")
                    imgs.append(rel + "/" + jpgs[0])
                    if root != d:
                        runs.append({"run": os.path.basename(root), "photos": len(jpgs)})
        imgs.sort(); runs.sort(key=lambda r: r["run"], reverse=True)
        out.append({"name": name, "display": name.replace("_", " "), "samples": int(len(db.people[name])),
                    "enrolled": int(db.meta[name]["enrolled"]), "learned": int(db.meta[name]["learned"]),
                    "photo": f"/file?p={quote('dataset/' + imgs[0])}" if imgs else "", "runs": runs})
    ds = os.path.join(HERE, "dataset")
    for name in sorted(os.listdir(ds)) if os.path.isdir(ds) else []:
        if os.path.isdir(os.path.join(ds, name)) and name not in db.people:
            out.append({"name": name, "display": name.replace("_", " "), "samples": 0, "enrolled": 0,
                        "learned": 0, "photo": "", "pending": True})
    return out


def events(n=200):
    p = os.path.join(HERE, "runs", "events.csv")
    if not os.path.exists(p):
        return []
    import csv
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[-n:]
    for r in rows:
        if r.get("snapshot"):
            r["snapshot_url"] = "/file?p=" + quote("runs/" + r["snapshot"].replace("\\", "/"))
    return rows[::-1]


def graphs():
    d = os.path.join(HERE, "runs")
    if not os.path.isdir(d):
        return []
    pngs = [f for f in os.listdir(d) if f.lower().endswith(".png") and not f.startswith("_")]
    pngs.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
    return [{"name": f[:-4], "url": "/file?p=" + quote("runs/" + f)} for f in pngs]


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>AVIS Dashboard</title>
<style>
:root{--bg:#0f1115;--card:#181b22;--line:#262b35;--txt:#e6e8ec;--dim:#8b93a3;--acc:#4ea1ff;--bad:#ff5d5d}
*{box-sizing:border-box}body{margin:0;font:14px system-ui,Segoe UI,sans-serif;background:var(--bg);color:var(--txt)}
header{display:flex;gap:6px;align-items:center;padding:14px 22px;border-bottom:1px solid var(--line)}
header b{font-size:18px;margin-right:18px}button.tab{background:none;border:0;color:var(--dim);padding:8px 12px;border-radius:8px;cursor:pointer;font-size:14px}
button.tab.on{background:var(--card);color:var(--txt)}main{padding:22px;max-width:1200px;margin:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.card img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px;background:#000}
.dim{color:var(--dim)}.btn{border:1px solid var(--line);background:#20242d;color:var(--txt);padding:7px 12px;border-radius:8px;cursor:pointer}
.btn.bad{border-color:#5a2a2a;color:var(--bad)}table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid var(--line);text-align:left}
td img{height:56px;border-radius:6px}textarea{width:100%;height:560px;background:#0b0d11;color:var(--txt);border:1px solid var(--line);border-radius:10px;padding:12px;font:13px Consolas,monospace}
.g img{width:100%;border-radius:8px;background:#fff}.tag{padding:2px 8px;border-radius:99px;font-size:12px}
.t-unknown{background:#3a2f12;color:#ffcf5d}.t-spoof{background:#3a1717;color:#ff7b7b}.t-seen{background:#13293d;color:#7cc0ff}.t-auto_lock{background:#2b2140;color:#c4a5ff}
#msg{margin-left:auto;color:var(--dim)}
</style></head><body>
<header><b>AVIS</b><button class="tab on" data-t="people">People</button><button class="tab" data-t="events">Events</button>
<button class="tab" data-t="graphs">Graphs &amp; reports</button><button class="tab" data-t="settings">Settings</button><span id="msg"></span></header>
<main id="view"></main>
<script>
const TOKEN="__TOKEN__", V=document.getElementById('view'), M=document.getElementById('msg');
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function j(u,o){const r=await fetch(u,o);if(!r.ok)throw new Error(await r.text());return r.json()}
const post=(u,b)=>j(u,{method:'POST',headers:{'X-Token':TOKEN,'Content-Type':'application/json'},body:JSON.stringify(b)});
const T={
 async people(){const p=await j('/api/people');V.innerHTML=p.length?'<div class="grid">'+p.map(x=>`<div class="card">
  ${x.photo?`<img src="${x.photo}">`:'<div style="aspect-ratio:1" class="dim">no photo</div>'}<h3 style="margin:10px 0 4px">${esc(x.display)}</h3>
  ${x.pending?'<div class="dim">Not enrolled yet - no face data stored</div>'
  :`<div class="dim">${x.samples} face samples &middot; ${x.enrolled} enrolled, ${x.learned} learned</div><br>
  ${(x.runs||[]).length?`<details style="margin-bottom:10px"><summary class="dim">${x.runs.length} photo folders (by run)</summary>${x.runs.map(r=>`<div class="dim">${esc(r.run)} &middot; ${r.photos}</div>`).join('')}</details>`:''}
  <button class="btn bad" onclick="forget('${esc(x.name)}','${esc(x.display)}')">Forget this person</button>`}</div>`).join('')+'</div>'
  :'<p class="dim">Nobody enrolled yet. Run Task &gt; AVIS: Enroll My Face.</p>'},
 async events(){const e=await j('/api/events');V.innerHTML=e.length?`<table><tr><th>Time</th><th>Event</th><th>Who</th><th>Match</th><th>Live</th><th>Snapshot</th></tr>`+
  e.map(r=>`<tr><td>${esc(r.time)}</td><td><span class="tag t-${esc(r.event)}">${esc(r.event)}</span></td><td>${esc((r.name||'').replace(/_/g,' '))}</td>
  <td>${esc(r.similarity)}</td><td>${esc(r.live_score)}</td><td>${r.snapshot_url?`<a href="${r.snapshot_url}" target="_blank"><img src="${r.snapshot_url}"></a>`:''}</td></tr>`).join('')+'</table>'
  :'<p class="dim">No events yet.</p>'},
 async graphs(){const [g,rep]=await Promise.all([j('/api/graphs'),j('/api/report')]);V.innerHTML=(rep.text?`<div class="card"><h3>Accuracy report</h3><pre>${esc(rep.text)}</pre></div><br>`:'')+
  (g.length?'<div class="grid" style="grid-template-columns:1fr">'+g.map(x=>`<div class="card g"><h3>${esc(x.name)}</h3><img src="${x.url}"></div>`).join('')+'</div>':'<p class="dim">No graphs yet. Run Task &gt; AVIS: Graph Last Run.</p>')},
 async settings(){const c=await j('/api/config');V.innerHTML=`<p class="dim">Edit and save. Restart AVIS to apply.</p><textarea id="cfg" spellcheck="false">${esc(c.text)}</textarea><br><br>
  <button class="btn" onclick="saveCfg()">Save settings</button>`}};
async function show(t){document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('on',b.dataset.t===t));try{await T[t]()}catch(e){V.innerHTML='<p style="color:var(--bad)">'+esc(e.message)+'</p>'}}
async function forget(n,d){if(!confirm('Permanently delete all face data for '+d+'? AVIS will treat them as Unknown.'))return;
 try{await post('/api/forget',{name:n});M.textContent=d+' forgotten';show('people')}catch(e){alert(e.message)}}
async function saveCfg(){try{await post('/api/config',{text:document.getElementById('cfg').value});M.textContent='Settings saved'}catch(e){alert(e.message)}}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>show(b.dataset.t));show('people');
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, PAGE.replace("__TOKEN__", TOKEN), "text/html; charset=utf-8")
        if u.path == "/api/people":
            return self._send(200, people())
        if u.path == "/api/events":
            return self._send(200, events())
        if u.path == "/api/graphs":
            return self._send(200, graphs())
        if u.path == "/api/report":
            p = os.path.join(HERE, "runs", "eval_report.txt")
            return self._send(200, {"text": open(p, encoding="utf-8").read() if os.path.exists(p) else ""})
        if u.path == "/api/config":
            return self._send(200, {"text": open(CONFIG, encoding="utf-8").read() if os.path.exists(CONFIG) else ""})
        if u.path == "/file":
            rel = parse_qs(u.query).get("p", [""])[0]
            full = os.path.realpath(os.path.join(HERE, rel))
            allowed = any(full.startswith(os.path.realpath(os.path.join(HERE, d)) + os.sep) for d in SERVE_DIRS)
            if not allowed or not os.path.isfile(full) or not full.lower().endswith((".jpg", ".jpeg", ".png")):
                return self._send(404, {"error": "not found"})
            ctype = "image/png" if full.lower().endswith(".png") else "image/jpeg"
            return self._send(200, open(full, "rb").read(), ctype)
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.headers.get("X-Token") != TOKEN:
            return self._send(403, {"error": "bad token - reload the page"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._send(400, {"error": "bad json"})
        path = urlparse(self.path).path
        if path == "/api/forget":
            name = str(body.get("name", ""))
            db = FaceDB(FACES)
            if name not in db.people or os.sep in name or "/" in name or name.startswith("."):
                return self._send(404, {"error": "no such person"})
            db.forget(name)
            shutil.rmtree(os.path.join(HERE, "dataset", name), ignore_errors=True)
            return self._send(200, {"ok": True})
        if path == "/api/config":
            text = str(body.get("text", ""))
            try:
                import yaml
                if not isinstance(yaml.safe_load(text) or {}, dict):
                    raise ValueError("top level must be sections")
            except Exception as e:
                return self._send(400, {"error": "Not saved - YAML error: " + html.escape(str(e))})
            with open(CONFIG, "w", encoding="utf-8") as f:
                f.write(text)
            return self._send(200, {"ok": True})
        self._send(404, {"error": "not found"})


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"AVIS dashboard: {url}  (Ctrl+C or kill the terminal to stop)")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
