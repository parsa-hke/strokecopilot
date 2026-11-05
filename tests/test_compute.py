"""Pod tooling (DEC-21): key scan, Vast client against a stub, pod lifecycle with stubs, remote scripts, pod_job helpers."""

import json
import subprocess

import pytest

from strokecopilot.compute import jobs, keyscan, pod, remote
from strokecopilot.compute.vast import FILTERS, Vast, VastError


# --- keyscan ----------------------------------------------------------------------------------------

def _repo(tmp_path, files):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    return tmp_path


def test_keyscan_finds_env_key_and_patterns(tmp_path):
    root = _repo(tmp_path, {"a.py": "x = 'abcdefabcdefabcdefabcdef1234'\n", "b.md": "sk-ant-api03-" + "A" * 40 + "\n", "c.txt": "fine\n"})
    found = keyscan.scan(root, {"VAST_API_KEY": "abcdefabcdefabcdefabcdef1234"})
    assert ("a.py", "the value of $VAST_API_KEY") in found
    assert any(f == "b.md" for f, _ in found)
    assert not any(f == "c.txt" for f, _ in found)
    with pytest.raises(keyscan.SecretInRepo) as e:
        keyscan.require_clean(root, {"VAST_API_KEY": "abcdefabcdefabcdefabcdef1234"})
    assert "abcdef" not in str(e.value)  # file names only, never the key


def test_keyscan_clean_repo(tmp_path):
    root = _repo(tmp_path, {"a.py": "print('hi')\n"})
    assert keyscan.scan(root, {"VAST_API_KEY": "zzzzzzzzzzzzzzzzzzzzzzzz"}) == []


def test_real_repo_is_clean():
    keyscan.require_clean(pod.REPO)


# --- Vast client ------------------------------------------------------------------------------------

class StubHttp:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, method, url, key, body=None):
        self.calls.append((method, url.split("/api", 1)[1], body))
        for (m, frag), answer in self.routes.items():
            if m == method and frag in url:
                return answer(body) if callable(answer) else answer
        raise VastError(f"no route for {method} {url}")


def test_vast_offers_sorted_and_filters_sent():
    http = StubHttp({("POST", "/bundles/"): {"offers": [{"id": 2, "dph_total": 0.45}, {"id": 1, "dph_total": 0.40}]}})
    v = Vast(key="k", http=http)
    assert [o["id"] for o in v.offers()] == [1, 2]
    body = http.calls[0][2]
    assert body["gpu_name"] == {"eq": "RTX 4090"} and body["type"] == "ondemand" and body["dph_total"] == {"lte": 0.50}
    assert "cuda_max_good" in FILTERS


def test_vast_create_and_refusal():
    v = Vast(key="k", http=StubHttp({("PUT", "/asks/7/"): {"success": True, "new_contract": 99}}))
    assert v.create(7, "img", 60, "lab") == 99
    v = Vast(key="k", http=StubHttp({("PUT", "/asks/7/"): {"success": False, "msg": "gone"}}))
    with pytest.raises(VastError):
        v.create(7, "img", 60, "lab")


def test_vast_wait_running_and_ssh():
    states = iter(["loading", "loading", "running"])
    inst = lambda _: {"instances": [{"id": 5, "actual_status": next(states), "public_ipaddr": "1.2.3.4 ",
                                      "ports": {"22/tcp": [{"HostPort": "40022"}]}, "ssh_host": "ssh5.vast.ai", "ssh_port": 1234}]}
    v = Vast(key="k", http=StubHttp({("GET", "/v1/instances"): inst}), sleep=lambda s: None)
    got = v.wait_running(5)
    assert Vast.ssh(got) == ("1.2.3.4", 40022, "root")
    assert Vast.ssh({"ssh_host": "ssh5.vast.ai", "ssh_port": 1234}) == ("ssh5.vast.ai", 1234, "root")


def test_vast_wait_running_gives_up():
    t = iter(range(0, 10_000, 100))
    v = Vast(key="k", http=StubHttp({("GET", "/v1/instances"): {"instances": [{"id": 5, "actual_status": "exited"}]}}),
             sleep=lambda s: None, clock=lambda: next(t))
    with pytest.raises(VastError):
        v.wait_running(5)


def test_vast_needs_key(monkeypatch):
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    with pytest.raises(VastError):
        Vast()


# --- remote scripts ---------------------------------------------------------------------------------

def test_watchdog_destroys_with_instance_key_then_stops():
    s = remote.start_watchdog(3.5 * 3600)
    assert "watchdog.sh 12600" in s and "setsid" in s
    assert "CONTAINER_API_KEY" in remote.WATCHDOG and "/proc/1/environ" in remote.WATCHDOG
    assert remote.WATCHDOG.index("-X DELETE") < remote.WATCHDOG.index("kill -TERM 1")
    assert "VAST_API_KEY" not in remote.WATCHDOG


def test_scripts_parse_in_bash(tmp_path):
    for i, script in enumerate([remote.BOOTSTRAP, remote.SETUP, remote.WATCHDOG, remote.start_watchdog(60),
                                remote.start_job("EXP-13", "python -c 'print(1)'"), remote.status("EXP-13")]):
        f = tmp_path / f"s{i}.sh"
        f.write_text(script)
        assert subprocess.run(["bash", "-n", str(f)]).returncode == 0, script


def test_start_job_quotes_and_records_exit():
    s = remote.start_job("EXP-13", "python -c 'print(1)'")
    assert "tmux new-session -d -s job" in s and "job.exit" in s and "/opt/conda/bin" in s


# --- jobs -------------------------------------------------------------------------------------------

def test_jobs_registry():
    for exp, spec in jobs.RUNS.items():
        assert spec["est_hours"] < spec["hours"]
        cmd = jobs.command(exp, "abc1234")
        assert "pod_job" in cmd and "--commit abc1234" in cmd and f"/root/sc/runs/{exp}" in cmd
    assert "--epochs 300" in jobs.get("EXP-14")["args"]
    with pytest.raises(KeyError):
        jobs.get("EXP-99")
    assert not any("tar" in p or "preprocessed" in p for p in jobs.COLLECT)


# --- pod lifecycle with a stub provider -------------------------------------------------------------

class StubProvider:
    def __init__(self, balance=5.0, offers=None):
        self._balance, self._offers = balance, offers if offers is not None else [{"id": 11, "dph_total": 0.40}]
        self.live, self.created, self.destroyed = {}, [], []

    def balance(self):
        return self._balance

    def offers(self, filters=None, limit=10):
        return self._offers

    def instances(self):
        return list(self.live.values())

    def get(self, i):
        return self.live.get(int(i))

    def create(self, offer_id, image, disk_gb, label):
        self.created.append((offer_id, image, disk_gb, label))
        self.live[77] = {"id": 77, "label": label, "actual_status": "running", "ssh_host": "h", "ssh_port": 22}
        return 77

    def wait_running(self, i):
        return self.live[int(i)]

    @staticmethod
    def ssh(inst):
        return inst["ssh_host"], inst["ssh_port"], "root"

    def destroy(self, i):
        self.destroyed.append(i)
        self.live.pop(int(i), None)

    def describe(self, o):
        return f"offer {o['id']} ${o['dph_total']}/h"


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    prov = StubProvider()
    monkeypatch.setattr(pod, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(pod, "provider", lambda name="vast": prov)
    monkeypatch.setattr(pod.keyscan, "require_clean", lambda root: None)
    monkeypatch.setattr(pod.time, "sleep", lambda s: None)
    tar = tmp_path / "isles24_slim_v1.tar"
    tar.write_bytes(b"x")
    monkeypatch.setattr(pod, "tarball_path", lambda: tar)
    monkeypatch.setattr(pod, "sha256", lambda p: pod.TARBALL_SHA256)
    monkeypatch.setattr(pod, "head_commit", lambda: "c0ffee1234")
    monkeypatch.setattr(pod, "uncommitted", lambda: [])
    monkeypatch.setattr(pod, "repo_archive", lambda commit, out: (out.parent.mkdir(parents=True, exist_ok=True), out.write_bytes(b"r"), out)[2])
    sent = []
    monkeypatch.setattr(pod, "wait_ssh", lambda state: None)
    monkeypatch.setattr(pod, "rsync_up", lambda state, local, remote_path: sent.append(("up", remote_path)))
    monkeypatch.setattr(pod, "rsync_down", lambda state, remote_path, local, inc: sent.append(("down", remote_path)))
    monkeypatch.setattr(pod, "drive_root", lambda: None)
    answers = {"status": "EXIT=0\n"}
    monkeypatch.setattr(pod, "remote_sh", lambda state, script, capture=True, timeout=600:
                        sent.append(("sh", script[:40])) or (answers["status"] if "job.exit ]" in script else "ok\n"))
    return prov, sent, answers


def test_up_refuses_without_go(stubbed):
    prov, _, _ = stubbed
    with pytest.raises(pod.PodError, match="--go"):
        pod.up("EXP-13", 11, go=False, log=lambda m: None)
    assert prov.created == []


def test_up_refuses_low_balance(stubbed):
    prov, _, _ = stubbed
    prov._balance = 0.1
    with pytest.raises(pod.PodError, match="Balance"):
        pod.up("EXP-13", 11, go=True, log=lambda m: None)
    assert prov.created == []


def test_up_refuses_unknown_offer(stubbed):
    prov, _, _ = stubbed
    with pytest.raises(pod.PodError, match="no longer"):
        pod.up("EXP-13", 999, go=True, log=lambda m: None)


def test_up_collect_down(stubbed):
    prov, sent, answers = stubbed
    state = pod.up("EXP-13", 11, go=True, log=lambda m: None)
    assert prov.created[0][3] == "strokecopilot-EXP-13" and state["instance_id"] == 77
    scripts = [s for kind, s in sent if kind == "sh"]
    assert any("watchdog" in s for s in scripts)
    assert ("up", "/root/sc/data/isles24_slim_v1.tar") in sent
    with pytest.raises(pod.PodError, match="already has a pod"):
        pod.up("EXP-13", 11, go=True, log=lambda m: None)
    answers["status"] = "RUNNING\n"
    with pytest.raises(pod.PodError, match="still running"):
        pod.collect("EXP-13", log=lambda m: None)
    answers["status"] = "EXIT=0\n"
    pod.collect("EXP-13", log=lambda m: None)
    assert prov.destroyed == [77] and prov.live == {}
    saved = json.loads((pod.RUNS / "EXP-13" / "pod.json").read_text())
    assert saved["destroyed"] and saved["cost_usd_estimate"] >= 0


def test_failed_setup_destroys_pod(stubbed, monkeypatch):
    prov, _, _ = stubbed
    monkeypatch.setattr(pod, "rsync_up", lambda *a: (_ for _ in ()).throw(pod.PodError("copy failed")))
    with pytest.raises(pod.PodError, match="copy failed"):
        pod.up("EXP-13", 11, go=True, log=lambda m: None)
    assert prov.destroyed == [77] and prov.live == {}


def test_down_raises_if_still_listed(stubbed):
    prov, _, _ = stubbed
    pod.up("EXP-13", 11, go=True, log=lambda m: None)
    prov.destroy = lambda i: None  # the destroy silently does nothing
    with pytest.raises(pod.PodError, match="still listed"):
        pod.down("EXP-13", log=lambda m: None)


# --- pod_job helpers --------------------------------------------------------------------------------

def test_effective_cpus_from_cgroup(tmp_path):
    from strokecopilot.vision.perfusion import pod_job
    (tmp_path / "cpu.max").write_text("1600000 100000\n")
    assert pod_job.effective_cpus(tmp_path) == 16
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert pod_job.effective_cpus(tmp_path) >= 1
    assert pod_job.workers(16) == (8, 14) and pod_job.workers(2) == (2, 2)


def test_gpu_summary_and_leak_check():
    from strokecopilot.vision.perfusion import pod_job
    s = pod_job.summarise_gpu([(0, 90.0, 9000.0), (10, 30.0, 9500.0), (20, 95.0, 9100.0)])
    assert s["util_median"] == 90.0 and s["share_below_50"] == 0.333 and s["mem_max_mib"] == 9500.0
    assert pod_job.summarise_gpu([]) is None
    folds = {"heldout": ["case-x"]}
    assert pod_job.leaked(["a case-x b", "clean"], folds) == 1 and pod_job.leaked(["clean"], folds) == 0


def test_offers_prefers_us_and_eu(stubbed):
    prov, _, _ = stubbed
    prov._offers = [{"id": 1, "dph_total": 0.30, "geolocation": "Henan, CN"}, {"id": 2, "dph_total": 0.40, "geolocation": "Texas, US"},
                    {"id": 3, "dph_total": 0.42, "geolocation": "Estonia, EE"}]
    assert [o["id"] for o in pod.offers("EXP-13", log=lambda m: None)] == [2, 3, 1]


def test_up_refuses_uncommitted_code(stubbed, monkeypatch):
    prov, _, _ = stubbed
    monkeypatch.setattr(pod, "uncommitted", lambda: ["src/strokecopilot/vision/perfusion/pod_job.py"])
    with pytest.raises(pod.PodError, match="Commit first"):
        pod.up("EXP-13", 11, go=True, log=lambda m: None)
    assert prov.created == []
