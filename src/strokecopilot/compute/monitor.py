"""A local progress page for a pod run: scripts/pod monitor EXP-14, then open http://localhost:8714.

Every `every` seconds it reads nnU-Net's training log from the pod over SSH (the lines with epoch, losses, pseudo Dice
and epoch time; never case IDs) and writes runs/EXP-<id>/monitor/data.json. index.html beside it redraws from that
file. When the pod is gone it writes a final snapshot (with our own Dice from metrics.json, once collected) and stops.
"""

import datetime as dt
import json
import re
import shlex
import time
from pathlib import Path

from strokecopilot.compute import jobs, pod

_STAMP = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+: "
_LINES = {
    "epoch": re.compile(_STAMP + r"Epoch (\d+)\s*$"),
    "train_loss": re.compile(_STAMP + r"train_loss (-?[\d.]+)"),
    "val_loss": re.compile(_STAMP + r"val_loss (-?[\d.]+)"),
    "dice": re.compile(_STAMP + r"Pseudo dice \[(?:np\.float32\()?(-?[\d.]+)"),
    "seconds": re.compile(_STAMP + r"Epoch time: ([\d.]+) s"),
}
EMA = 0.9  # nnU-Net's smoothing of the pseudo Dice: ema = 0.9 * ema + 0.1 * dice


def parse_log(text):
    """[{epoch, train_loss, val_loss, dice, ema, seconds, ended}] for every finished epoch in a training log."""
    out, cur, ema = [], None, None
    for line in text.splitlines():
        for key, pattern in _LINES.items():
            m = pattern.match(line.strip())
            if not m:
                continue
            if key == "epoch":
                cur = {"epoch": int(m.group(2))}
            elif cur is not None:
                cur[key] = float(m.group(2))
                if key == "seconds":
                    cur["ended"] = m.group(1)
                    if "dice" in cur:
                        ema = cur["dice"] if ema is None else EMA * ema + (1 - EMA) * cur["dice"]
                        cur["ema"] = round(ema, 4)
                    out.append(cur)
                    cur = None
            break
    return out


# Every training log, oldest first: a resumed run (EXP-14, after Vast stopped the host) starts a new one.
REMOTE = ("t=$(ls -tr {run}/nnUNet_results/*/*/fold_*/training_log_*.txt 2>/dev/null); "
          "[ -n \"$t\" ] && cat $t | grep -E 'Epoch [0-9]+ *$|train_loss|val_loss|Pseudo dice|Epoch time'; "
          "echo '#EXIT' $(cat {run}/job.exit 2>/dev/null); grep -h 'armed for' /root/sc/watchdog.log 2>/dev/null | tail -1")


def summarise(exp, state, log_text, now=None, final_metrics=None):
    """The data.json content from the pod state, the remote output and the clock."""
    now = now or dt.datetime.now(dt.timezone.utc)
    spec = jobs.get(exp)
    m = re.search(r"--epochs (\d+)", spec["args"])
    total = int(m.group(1)) if m else None
    epochs = parse_log(log_text)
    exit_code = None
    em = re.search(r"#EXIT (\S+)", log_text)
    if em:
        exit_code = int(em.group(1))
    armed = re.search(r"armed for (\d+) s at (\S+)", log_text)
    deadline = None
    if armed:
        deadline = (dt.datetime.fromisoformat(armed.group(2).replace("Z", "+00:00")) + dt.timedelta(seconds=int(armed.group(1)))).isoformat()
    created = dt.datetime.fromisoformat(state["created_utc"])
    end = dt.datetime.fromisoformat(state["destroyed_utc"]) if state.get("destroyed_utc") else now
    hours = (end - created).total_seconds() / 3600
    recent = [e["seconds"] for e in epochs[-20:]]
    per_epoch = sorted(recent)[len(recent) // 2] if recent else None
    left = None if total is None or per_epoch is None else max(0, total - len(epochs))
    eta = None if left is None else (now + dt.timedelta(seconds=left * per_epoch + 300)).isoformat()
    projected = None if left is None else round((hours + (left * per_epoch + 600) / 3600) * state["dph"], 2)
    if state.get("destroyed"):
        phase = "finished, pod destroyed"
    elif exit_code is not None:
        phase = "job ended (exit %d); collecting" % exit_code
    elif not epochs:
        phase = "setting up (upload, install, preprocessing)"
    elif left == 0:
        phase = "validating and scoring"
    else:
        phase = "training"
    return {
        "exp": exp, "what": spec["what"], "phase": phase, "updated_utc": now.isoformat(), "instance": state.get("instance_id"),
        "price_per_hour": state["dph"], "hours_up": round(hours, 2), "cost_so_far": round(hours * state["dph"], 2),
        "cost_projected": projected, "hour_limit": state.get("hours_limit"), "watchdog_deadline_utc": deadline,
        "epochs_total": total, "epochs_done": len(epochs), "seconds_per_epoch": per_epoch, "eta_utc": eta,
        "best_ema": max((e.get("ema", 0) for e in epochs), default=None), "last": epochs[-1] if epochs else None,
        "epochs": [{k: e.get(k) for k in ("epoch", "dice", "ema", "seconds", "train_loss", "val_loss")} for e in epochs],
        "exit_code": exit_code, "final": final_metrics,
    }


def _final_metrics(exp):
    p = pod.RUNS / exp / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    fold = str(m.get("fold", 0))
    s = (m.get("summary") or {}).get("per_fold", {}).get(fold, {})
    pick = lambda k: {"mean": s[k]["mean"], "std": s[k].get("std"), "n": s[k].get("n")} if isinstance(s.get(k), dict) else None
    return {"dice": pick("dice"), "dice_reliable": pick("dice_reliable"), "abs_error_ml": pick("abs_error_ml")}


def run(exp, every=60, log=print):
    out = pod.RUNS / exp / "monitor"
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(PAGE)
    last_text = ""
    while True:
        state = pod.load_state(exp)
        if not state:
            raise pod.PodError(f"{exp}: no pod.json.")
        if not state.get("destroyed"):
            try:
                run_dir = f"{pod.remote.ROOT}/runs/{exp}"
                last_text = pod.remote_sh(state, REMOTE.replace("{run}", run_dir), timeout=60)
            except Exception as e:  # SSH can drop (VPN); keep the last good snapshot and try again
                log(f"{dt.datetime.now(dt.timezone.utc):%H:%M} could not reach the pod: {str(e)[:120]}")
        data = summarise(exp, state, last_text, final_metrics=_final_metrics(exp))
        tmp = out / "data.json.tmp"
        tmp.write_text(json.dumps(data))
        tmp.replace(out / "data.json")
        if state.get("destroyed"):
            log(f"{exp}: pod destroyed; final snapshot written.")
            return data
        time.sleep(every)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pod run monitor</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e4e4df;--accent:#2f6fde;--accent2:#c9d9f7;--ok:#1f8a4c;--warn:#b5651d;--bad:#c0392b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121212;--card:#1c1c1c;--ink:#ececea;--muted:#9a9a95;--line:#2c2c2a;--accent:#7aa7ff;--accent2:#2a3b5c;--ok:#4cc38a;--warn:#e0a45a;--bad:#ff7a6b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:960px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 20px}
.phase{display:inline-block;padding:2px 10px;border-radius:999px;background:var(--accent2);color:var(--ink);font-size:13px;margin-left:8px;vertical-align:middle}
.bar{height:14px;background:var(--line);border-radius:7px;overflow:hidden;margin:8px 0 4px}.bar>div{height:100%;background:var(--accent);transition:width .6s}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}.tile .v{font-size:22px;font-weight:600;margin-top:2px;font-variant-numeric:tabular-nums}
.tile .s{color:var(--muted);font-size:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-top:12px}
.card h2{font-size:15px;margin:0 0 8px}svg{width:100%;height:auto;display:block}
.legend{display:flex;gap:16px;color:var(--muted);font-size:12px;margin-top:6px}.sw{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:6px}
.final{border-color:var(--ok)}.final .v{color:var(--ok)}.foot{color:var(--muted);font-size:12px;margin-top:18px}
.stale{color:var(--warn)}
</style></head><body><main>
<h1 id="title">Pod run</h1><p class="sub" id="what"></p>
<div><strong id="pct">–</strong> <span id="phase" class="phase">loading</span></div>
<div class="bar"><div id="fill" style="width:0"></div></div><div class="sub" id="eta"></div>
<div class="grid" id="tiles"></div>
<div class="card" id="finalcard" hidden><h2>Result (our scoring, fold validation)</h2><div class="grid" id="final" style="margin:0"></div></div>
<div class="card"><h2>Validation pseudo Dice per epoch</h2><svg id="chart" viewBox="0 0 900 260" role="img" aria-label="Pseudo Dice per epoch"></svg>
<div class="legend"><span><i class="sw" style="background:var(--accent2)"></i>pseudo Dice (each epoch)</span><span><i class="sw" style="background:var(--accent)"></i>EMA (what nnU-Net uses to pick the best checkpoint)</span></div>
<p class="sub" style="margin:8px 0 0;font-size:12px">Pseudo Dice is nnU-Net's quick estimate on random validation patches, not the final score. The final Dice comes from our own scoring after training.</p></div>
<p class="foot" id="foot"></p>
</main><script>
const $=id=>document.getElementById(id);
const fmtT=s=>s?new Date(s).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})+' local ('+new Date(s).toISOString().slice(11,16)+' UTC)':'–';
const usd=x=>x==null?'–':'$'+x.toFixed(2);
function tile(k,v,s){return `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s||''}</div></div>`}
function chart(ep){const W=900,H=260,P=36,svg=$('chart');if(!ep.length){svg.innerHTML='';return}
 const n=Math.max(ep.length, 2), xs=i=>P+(W-P-10)*i/(n-1), vals=ep.map(e=>e.dice).filter(v=>v!=null), hi=Math.max(0.8,...vals), ys=v=>H-P-(H-P-10)*v/hi;
 let g='';for(let t=0;t<=hi+1e-9;t+=0.2){g+=`<line x1="${P}" x2="${W-10}" y1="${ys(t)}" y2="${ys(t)}" stroke="var(--line)"/><text x="${P-6}" y="${ys(t)+4}" font-size="11" text-anchor="end" fill="var(--muted)">${t.toFixed(1)}</text>`}
 const line=(k,c,w)=>`<polyline fill="none" stroke="${c}" stroke-width="${w}" points="${ep.map((e,i)=>e[k]==null?'':xs(i)+','+ys(e[k])).join(' ')}"/>`;
 g+=line('dice','var(--accent2)',1.5)+line('ema','var(--accent)',2.5);
 g+=`<text x="${W-10}" y="${H-10}" font-size="11" text-anchor="end" fill="var(--muted)">epoch ${ep[ep.length-1].epoch}</text><text x="${P}" y="${H-10}" font-size="11" fill="var(--muted)">epoch 0</text>`;
 svg.innerHTML=g}
async function tick(){let d;try{d=await (await fetch('data.json?'+Date.now())).json()}catch(e){$('foot').textContent='Waiting for data.json…';return}
 document.title=`${d.exp} · ${d.epochs_done}/${d.epochs_total||'?'}`;$('title').textContent=d.exp+' monitor';$('what').textContent=d.what+' · instance '+d.instance;
 const pct=d.epochs_total?Math.min(100,100*d.epochs_done/d.epochs_total):0;$('pct').textContent=`Epoch ${d.epochs_done} of ${d.epochs_total} (${pct.toFixed(0)}%)`;$('fill').style.width=pct+'%';$('phase').textContent=d.phase;
 $('eta').textContent=d.eta_utc&&!d.final?'Expected to finish around '+fmtT(d.eta_utc):'';
 const L=d.last||{};$('tiles').innerHTML=tile('Per epoch',d.seconds_per_epoch?d.seconds_per_epoch.toFixed(1)+' s':'–','median of the last 20')+tile('Latest pseudo Dice',L.dice!=null?L.dice.toFixed(3):'–','EMA '+(L.ema!=null?L.ema.toFixed(3):'–')+' · best EMA '+(d.best_ema?d.best_ema.toFixed(3):'–'))
 +tile('Cost so far',usd(d.cost_so_far),`${d.hours_up} h at $${d.price_per_hour.toFixed(3)}/h`)+tile('Projected total',usd(d.cost_projected),'if the pace holds')+tile('Watchdog',d.watchdog_deadline_utc?new Date(d.watchdog_deadline_utc).toISOString().slice(11,16)+' UTC':'–',`destroys the pod at the ${d.hour_limit} h limit`);
 chart(d.epochs||[]);
 if(d.final&&d.final.dice){$('finalcard').hidden=false;const f=d.final;$('final').innerHTML=tile('Dice',f.dice.mean.toFixed(3),'± '+(f.dice.std??0).toFixed(3)+' over '+f.dice.n+' cases')+(f.dice_reliable?tile('Dice, reliable Tmax',f.dice_reliable.mean.toFixed(3),''):'')+(f.abs_error_ml?tile('Abs volume error',f.abs_error_ml.mean.toFixed(1)+' mL',''):'');$('finalcard').classList.add('final')}
 const age=(Date.now()-new Date(d.updated_utc))/1000;$('foot').innerHTML=`Updated ${Math.round(age)} s ago`+(age>300?' <span class="stale">(stale: is the monitor running and the VPN on?)</span>':'')+' · refreshes every 30 s';}
tick();setInterval(tick,30000);
</script></body></html>
"""
