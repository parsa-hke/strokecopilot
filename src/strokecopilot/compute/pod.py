"""The pod lifecycle: up, status, collect, down (DEC-21, plan §4.6). Entry point: scripts/pod.

    scripts/pod offers EXP-13                    # search only: the cheapest offers that pass the filters
    scripts/pod up EXP-13 --offer 123 --go       # create (after Parsa approves the offer, price and hour limit)
    scripts/pod status EXP-13                    # job state, last log lines, watchdog
    scripts/pod collect EXP-13                   # results back to the Mac and Drive, then down
    scripts/pod down EXP-13                      # destroy and confirm by listing instances

State lives in runs/EXP-<id>/pod.json on the Mac (runs/ is never committed). No token is ever sent to the pod:
the tarball and an archive of the repo at the pinned commit go up over SSH, and results come back the same way.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from strokecopilot.compute import jobs, keyscan, remote

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "runs"
SSH_KEY = Path(os.environ.get("POD_SSH_KEY", "~/.ssh/vast_pod")).expanduser()
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"
DISK_GB = 60
BALANCE_MARGIN = 1.5
TARBALL = "isles24_slim_v1.tar"
TARBALL_SHA256 = "b0afd16e4cbfb16a69fb5190083fc3869f5dcc6c92f86f10135b0de1f59f4728"


class PodError(RuntimeError):
    pass


def provider(name="vast"):
    if name == "vast":
        from strokecopilot.compute.vast import Vast
        return Vast()
    if name == "runpod":
        from strokecopilot.compute.runpod import RunPod
        return RunPod()
    raise PodError(f"Unknown provider {name!r}")


def state_path(exp):
    return RUNS / exp / "pod.json"


def load_state(exp):
    p = state_path(exp)
    return json.loads(p.read_text()) if p.exists() else None


def save_state(exp, state):
    state_path(exp).parent.mkdir(parents=True, exist_ok=True)
    state_path(exp).write_text(json.dumps(state, indent=1))


def label(exp):
    return f"strokecopilot-{exp}"


# --- local helpers --------------------------------------------------------------------------------

def drive_root():
    """The StrokeCopilot folder of Google Drive for desktop on this Mac, or None."""
    base = Path("~/Library/CloudStorage").expanduser()
    for d in sorted(base.glob("GoogleDrive-*")) if base.exists() else []:
        for sub in ("My Drive/StrokeCopilot", "MyDrive/StrokeCopilot"):
            if (d / sub).is_dir():
                return d / sub
    return None


def tarball_path():
    local = REPO / "data" / TARBALL
    if local.exists():
        return local
    root = drive_root()
    if root and (root / "slim" / TARBALL).exists():
        return root / "slim" / TARBALL
    raise PodError(f"{TARBALL} is neither in data/ nor in Drive for desktop. Install Drive for desktop or download it to data/.")


def sha256(path, chunk=16 * 2**20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def head_commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def uncommitted(paths=("src", "pyproject.toml", "schemas", "splits", ":!src/strokecopilot/compute")):
    """Files under `paths` that differ from HEAD or are untracked: the pod runs HEAD, so these would be missing.
    compute/ is left out: it runs on the Mac only."""
    out = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [line[3:] for line in out.splitlines() if line.strip()]


def repo_archive(commit, out):
    subprocess.run(["git", "archive", "--format=tar", "-o", str(out), commit], cwd=REPO, check=True)
    return out


def ssh_base(host, port, user):
    return ["ssh", "-i", str(SSH_KEY), "-p", str(port), "-o", "StrictHostKeyChecking=accept-new",
            "-o", "UserKnownHostsFile=" + str(RUNS / ".pod_known_hosts"), "-o", "ServerAliveInterval=30",
            "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", f"{user}@{host}"]


def run(cmd, check=True, capture=False, timeout=None):
    out = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
    if check and out.returncode:
        raise PodError(f"Command failed ({out.returncode}): {' '.join(cmd[:3])} … {(out.stderr or '')[-400:]}")
    return out


def remote_sh(state, script, capture=True, timeout=600):
    cmd = ssh_base(state["host"], state["port"], state["user"]) + ["bash -s"]
    out = subprocess.run(cmd, input=script, capture_output=capture, text=True, timeout=timeout)
    if out.returncode:
        raise PodError(f"Remote script failed ({out.returncode}): {(out.stderr or out.stdout or '')[-600:]}")
    return out.stdout


def rsync_up(state, local, remote_path):
    e = " ".join(shlex.quote(x) for x in ssh_base(state["host"], state["port"], state["user"])[:-1])
    run(["rsync", "-a", "--partial", "-e", e, str(local), f"{state['user']}@{state['host']}:{remote_path}"])


def rsync_down(state, remote_path, local, includes):
    e = " ".join(shlex.quote(x) for x in ssh_base(state["host"], state["port"], state["user"])[:-1])
    filters = []
    for pattern in includes:
        filters += ["--include", pattern]
    filters += ["--include", "*/", "--exclude", "*"]
    run(["rsync", "-a", "--prune-empty-dirs", *filters, "-e", e, f"{state['user']}@{state['host']}:{remote_path}", str(local)])


def wait_ssh(state, tries=8, pause=15):
    for _ in range(tries):
        out = subprocess.run(ssh_base(state["host"], state["port"], state["user"]) + ["echo ok"], capture_output=True, text=True)
        if out.returncode == 0 and "ok" in out.stdout:
            return
        time.sleep(pause)
    raise PodError("SSH never answered.")


# --- the verbs ------------------------------------------------------------------------------------

# DEC-21 prefers hosts in the US or EU; Vast's geolocation ends with the country code ("Texas, US").
PREFERRED = {"US", "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT", "LV", "LT",
             "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE"}


def preferred(offer):
    return (offer.get("geolocation") or "").rsplit(",", 1)[-1].strip().upper() in PREFERRED


def search(p, rounds=6):
    """Every offer seen over several searches, cheapest first. Vast answers each search with a varying sample of
    about 15 matching offers (checked 5 Nov 2026), so one search can miss an offer that is still for rent."""
    seen = {}
    for _ in range(rounds):
        for o in p.offers(limit=64):
            seen[str(o["id"])] = o
    return sorted(seen.values(), key=lambda o: o.get("dph_total", 1e9))


def offers(exp, provider_name="vast", log=print, show=6):
    """The cheapest offers that pass the filters, US and EU hosts first."""
    spec = jobs.get(exp)
    p = provider(provider_name)
    found = search(p)
    found = ([o for o in found if preferred(o)] + [o for o in found if not preferred(o)])[:show]
    balance = p.balance()
    log(f"{exp}: {spec['what']}; hour limit {spec['hours']} h; estimated {spec['est_hours']} h. Balance ${balance:.2f}.")
    if not found:
        log("No offer passes the DEC-21 filters right now.")
    for o in found:
        log("  " + p.describe(o) + f"  -> estimated ${o['dph_total'] * spec['est_hours']:.2f}, at most ${o['dph_total'] * spec['hours']:.2f}")
    return found


def up(exp, offer_id, go=False, provider_name="vast", commit=None, log=print):
    """Create the pod and start the job. Refuses without --go, with a secret in the repo, with a pod already up for
    this run, or when the balance does not cover the run's estimated cost with a margin."""
    spec = jobs.get(exp)
    keyscan.require_clean(REPO)
    if load_state(exp) and load_state(exp).get("instance_id") and not load_state(exp).get("destroyed"):
        raise PodError(f"{exp} already has a pod ({load_state(exp)['instance_id']}). Run collect or down first.")
    p = provider(provider_name)
    if any(i.get("label") == label(exp) for i in p.instances()):
        raise PodError(f"An instance labelled {label(exp)} already exists. Run down first.")
    offer = next((o for o in search(p, rounds=10) if str(o["id"]) == str(offer_id)), None)
    if offer is None:
        raise PodError(f"Offer {offer_id} no longer passes the filters or is gone. Run offers again.")
    need = offer["dph_total"] * spec["est_hours"] * BALANCE_MARGIN
    if p.balance() < need:
        raise PodError(f"Balance ${p.balance():.2f} is under ${need:.2f} (1.5 x the estimated cost). Top up first.")
    if not go:
        raise PodError(f"Not created: add --go once Parsa has approved {p.describe(offer)} with an hour limit of {spec['hours']} h.")
    dirty = uncommitted()
    if dirty and commit is None:
        raise PodError(f"{len(dirty)} file(s) under src/ or the package config are not committed, and the pod runs the "
                       f"committed code: {', '.join(dirty[:5])}. Commit first.")
    tar = tarball_path()
    if sha256(tar) != TARBALL_SHA256:
        raise PodError("The local tarball's sha256 does not match the recorded one.")
    commit = commit or head_commit()
    instance_id = p.create(offer["id"], IMAGE, DISK_GB, label(exp))
    state = {"exp": exp, "provider": provider_name, "instance_id": instance_id, "offer": offer["id"],
             "dph": offer["dph_total"], "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "commit": commit,
             "hours_limit": spec["hours"], "destroyed": False}
    save_state(exp, state)
    log(f"Created instance {instance_id} ({p.describe(offer)}). Waiting for it to run.")
    try:
        inst = p.wait_running(instance_id)
        state["host"], state["port"], state["user"] = p.ssh(inst)
        save_state(exp, state)
        wait_ssh(state)
        log(remote_sh(state, remote.BOOTSTRAP).strip())
        log(remote_sh(state, remote.start_watchdog(spec["hours"] * 3600)).strip())
        started = time.time()
        rsync_up(state, tar, f"{remote.ROOT}/data/{TARBALL}")
        archive = repo_archive(commit, RUNS / exp / "repo.tar")
        rsync_up(state, archive, f"{remote.ROOT}/repo.tar")
        archive.unlink(missing_ok=True)
        log(f"Copied the tarball and the repo at {commit[:7]} in {time.time() - started:.0f} s.")
        remote_sh(state, "cd /root/sc && tar -xf repo.tar -C repo && rm repo.tar")
        log(remote_sh(state, remote.SETUP.replace("{nnunet}", jobs.NNUNET_VERSION), timeout=1800).strip())
        log(remote_sh(state, remote.start_job(exp, jobs.command(exp, commit))).strip())
    except Exception:
        log("Setup failed; destroying the pod so nothing is left running.")
        down(exp, log=log)
        raise
    state["job_started_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    save_state(exp, state)
    log(f"{exp} is running on {state['host']}:{state['port']}. Watch it with: scripts/pod status {exp}")
    return state


def status(exp, log=print):
    state = load_state(exp)
    if not state or state.get("destroyed"):
        log(f"{exp}: no pod.")
        return None
    out = remote_sh(state, remote.status(exp), timeout=60)
    log(out.rstrip())
    return out


def collect(exp, force=False, log=print):
    """Copy the run's outputs back (config, metrics, per-case scores, best and final checkpoint, logs), put them in
    Drive, then destroy the pod. Refuses while the job is still running unless force."""
    state = load_state(exp)
    if not state or state.get("destroyed"):
        raise PodError(f"{exp}: no pod to collect from.")
    out = remote_sh(state, remote.status(exp), timeout=60)
    if "EXIT=" not in out and not force:
        raise PodError(f"{exp} is still running. Use --force to collect anyway (the job is then lost).")
    local = RUNS / exp
    rsync_down(state, f"{remote.ROOT}/runs/{exp}/", local, jobs.COLLECT)
    rsync_down(state, f"{remote.ROOT}/", local / "pod_logs", ["watchdog.log", "pip.log"])
    root = drive_root()
    if root:
        dest = root / "runs" / jobs.get(exp)["run_dir"]
        run(["rsync", "-a", "--exclude", "pod.json", "--exclude", "repo.tar", str(local) + "/", str(dest) + "/"])
        log(f"Copied to Drive: {dest}")
    else:
        log("Drive for desktop not found: outputs are in runs/ on this Mac only.")
    down(exp, log=log)
    return local


def down(exp, log=print):
    """Destroy the pod and confirm it is gone by listing instances. Records the measured cost."""
    state = load_state(exp)
    if not state or not state.get("instance_id"):
        log(f"{exp}: no pod recorded.")
        return
    p = provider(state["provider"])
    if p.get(state["instance_id"]) is not None:
        p.destroy(state["instance_id"])
    for _ in range(12):
        if p.get(state["instance_id"]) is None and not any(i.get("label") == label(exp) for i in p.instances()):
            break
        time.sleep(10)
    else:
        raise PodError(f"Instance {state['instance_id']} is still listed after destroy. Check the console now.")
    created = dt.datetime.fromisoformat(state["created_utc"])
    hours = (dt.datetime.now(dt.timezone.utc) - created).total_seconds() / 3600
    state.update(destroyed=True, destroyed_utc=dt.datetime.now(dt.timezone.utc).isoformat(), hours_up=round(hours, 3),
                 cost_usd_estimate=round(hours * state["dph"], 3))
    save_state(exp, state)
    log(f"{exp}: instance {state['instance_id']} destroyed and gone from the list. Up {hours:.2f} h, about ${hours * state['dph']:.2f}.")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="scripts/pod", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verb", choices=["offers", "up", "status", "collect", "down"])
    ap.add_argument("exp")
    ap.add_argument("--offer")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--provider", default="vast")
    a = ap.parse_args(argv)
    try:
        if a.verb == "offers":
            offers(a.exp, a.provider)
        elif a.verb == "up":
            if not a.offer:
                raise PodError("up needs --offer <id> from `scripts/pod offers`.")
            up(a.exp, a.offer, a.go, a.provider)
        elif a.verb == "status":
            status(a.exp)
        elif a.verb == "collect":
            collect(a.exp, a.force)
        else:
            down(a.exp)
    except (PodError, keyscan.SecretInRepo) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
