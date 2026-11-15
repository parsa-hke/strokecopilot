"""Pod tooling (DEC-21): key scan, Vast client against a stub, pod lifecycle with stubs, remote scripts, pod_job helpers."""

import json
import subprocess
import time

import pytest

from strokecopilot.compute import jobs, keyscan, pod, remote
from strokecopilot.compute import vast
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


def test_keyscan_finds_the_open_model_keys(tmp_path):
    """The keys VLLMJSON and the Hugging Face tools read are checked like the API keys."""
    root = _repo(tmp_path, {"cfg.json": '{"k": "vllmkeyvllmkeyvllmkey"}\n', "n.md": "hfvaluehfvaluehfvalue\n"})
    found = keyscan.scan(root, {"VLLM_API_KEY": "vllmkeyvllmkeyvllmkey", "HF_TOKEN": "hfvaluehfvaluehfvalue"})
    assert ("cfg.json", "the value of $VLLM_API_KEY") in found and ("n.md", "the value of $HF_TOKEN") in found


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
        if "serve" in spec:  # a serve pod runs vLLM only, with no job of ours
            continue
        cmd = jobs.command(exp, "abc1234")
        assert "pod_job" in cmd and "--commit abc1234" in cmd and f"/root/sc/runs/{exp}" in cmd
    assert "--epochs 300" in jobs.get("EXP-14")["args"]
    with pytest.raises(KeyError):
        jobs.get("EXP-99")
    assert not any("tar" in p or "preprocessed" in p for p in jobs.COLLECT)


def test_fold_jobs_and_the_carried_predict_run():
    for k, exp in enumerate(("EXP-18", "EXP-19", "EXP-20", "EXP-21"), start=1):
        spec = jobs.get(exp)
        assert f"--mode train --fold {k} --epochs 300" == spec["args"] and spec["run_dir"] == f"{exp}_nnunet_fold{k}"
    assert jobs.chain("EXP-18") == ["EXP-22", "EXP-18"] and jobs.chain("EXP-19") == ["EXP-19"]
    cmd = jobs.command("EXP-18", "abc1234")
    first, second = cmd.split("; python")
    assert "--exp EXP-22 --mode predict --fold 0" in first and "> /root/sc/runs/EXP-22/job.log" in first
    assert "echo $? > /root/sc/runs/EXP-22/job.exit" in first
    assert "--exp EXP-18 --mode train --fold 1" in second and "EXP-22" not in second
    assert jobs.inputs("EXP-18") == {jobs.EXP14_CHECKPOINT: "/root/sc/inputs/EXP-22/checkpoint_final.pth"}
    assert jobs.inputs("EXP-19") == {}
    assert "--checkpoint /root/sc/inputs/EXP-22/checkpoint_final.pth" in jobs.get("EXP-22")["args"]


# --- pod lifecycle with a stub provider -------------------------------------------------------------

class StubProvider:
    def __init__(self, balance=5.0, offers=None):
        self._balance, self._offers = balance, offers if offers is not None else [{"id": 11, "dph_total": 0.40, "duration": 500 * 3600}]
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
    monkeypatch.setattr(pod, "rsync_up", lambda state, local, remote_path, timeout=None: sent.append(("up", remote_path, timeout)[:2]))
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


def test_up_sends_inputs_and_collect_brings_every_run(stubbed, tmp_path, monkeypatch):
    prov, sent, answers = stubbed
    drive = tmp_path / "drive"
    monkeypatch.setattr(pod, "drive_root", lambda: drive)
    with pytest.raises(pod.PodError, match="not on Drive"):
        pod.up("EXP-18", 11, go=True, log=lambda m: None)
    assert prov.created == []
    ckpt = drive / jobs.EXP14_CHECKPOINT
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"w")
    monkeypatch.setattr(pod, "run", lambda cmd, **k: sent.append(("drive", cmd[-1])))
    state = pod.up("EXP-18", 11, go=True, log=lambda m: None)
    assert ("up", "/root/sc/inputs/EXP-22/checkpoint_final.pth") in sent and state["runs"] == ["EXP-22", "EXP-18"]
    assert list(state["inputs_sha256"]) == ["/root/sc/inputs/EXP-22/checkpoint_final.pth"]
    pod.collect("EXP-18", log=lambda m: None)
    assert ("down", "/root/sc/runs/EXP-22/") in sent and ("down", "/root/sc/runs/EXP-18/") in sent
    drive_dests = [d for kind, d in sent if kind == "drive"]
    assert any(d.endswith("EXP-22_nnunet_fold0_masks/") for d in drive_dests)
    assert any(d.endswith("EXP-18_nnunet_fold1/") for d in drive_dests)
    assert prov.destroyed == [77]


def test_tarball_upload_has_a_time_limit(stubbed, monkeypatch):
    prov, sent, _ = stubbed
    limits = {}
    monkeypatch.setattr(pod, "rsync_up", lambda state, local, remote_path, timeout=None: limits.__setitem__(remote_path, timeout))
    state = pod.up("EXP-13", 11, go=True, log=lambda m: None)
    assert limits["/root/sc/data/isles24_slim_v1.tar"] == pod.UPLOAD_LIMIT_S and state["machine_id"] is None


def test_failed_setup_destroys_pod(stubbed, monkeypatch):
    prov, _, _ = stubbed
    monkeypatch.setattr(pod, "rsync_up", lambda *a, **k: (_ for _ in ()).throw(pod.PodError("copy failed")))
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

def test_pod_job_predict_arguments_are_checked(tmp_path):
    from strokecopilot.vision.perfusion import pod_job
    base = ["--exp", "EXP-22", "--commit", "abc", "--tar", "t", "--run-dir", str(tmp_path / "r"), "--work", str(tmp_path / "w")]
    with pytest.raises(SystemExit, match="needs --checkpoint"):
        pod_job.main(base + ["--mode", "predict"])
    with pytest.raises(SystemExit, match="predict only"):
        pod_job.main(base + ["--mode", "train", "--epochs", "3", "--limit", "2"])
    with pytest.raises(SystemExit, match="predict only"):
        pod_job.main(base + ["--mode", "dummy", "--device", "cpu"])


def test_gpu_name_is_empty_without_nvidia_smi(monkeypatch):
    from strokecopilot.vision.perfusion import pod_job
    monkeypatch.setattr(pod_job.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi")))
    assert pod_job.gpu_name() == ""


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


def test_search_unions_varying_samples():
    class Sampler:
        answers = iter([[{"id": 1, "dph_total": 0.4}], [{"id": 2, "dph_total": 0.3}], [{"id": 1, "dph_total": 0.4}]])

        def offers(self, filters=None, limit=10):
            return next(self.answers, [])
    assert [o["id"] for o in pod.search(Sampler(), rounds=4)] == [2, 1]


def test_bootstrap_installs_a_compiler():
    assert "build-essential" in remote.BOOTSTRAP and "command -v gcc" in remote.BOOTSTRAP


LOG = """2026-11-05 15:04:28.141622: Epoch 0
2026-11-05 15:04:28.141860: Current learning rate: 0.01
2026-11-05 15:05:37.150419: train_loss -0.1016
2026-11-05 15:05:37.150601: val_loss -0.1722
2026-11-05 15:05:37.150646: Pseudo dice [np.float32(0.3117)]
2026-11-05 15:05:37.150700: Epoch time: 69.01 s
2026-11-05 15:05:38.035856: Epoch 1
2026-11-05 15:06:15.382176: train_loss -0.1465
2026-11-05 15:06:15.382359: val_loss -0.2016
2026-11-05 15:06:15.382405: Pseudo dice [np.float32(0.369)]
2026-11-05 15:06:15.382456: Epoch time: 37.35 s
2026-11-05 15:06:16.000000: Epoch 2
#EXIT
2026-11-05T15:14:26Z watchdog armed for 19800 s at 2026-11-05T15:14:26Z
"""


def test_monitor_parses_finished_epochs_only():
    from strokecopilot.compute import monitor
    ep = monitor.parse_log(LOG)
    assert [e["epoch"] for e in ep] == [0, 1]
    assert ep[1]["dice"] == 0.369 and ep[1]["seconds"] == 37.35 and ep[1]["train_loss"] == -0.1465
    assert ep[0]["ema"] == 0.3117 and ep[1]["ema"] == round(0.9 * 0.3117 + 0.1 * 0.369, 4)


def test_monitor_summary_eta_cost_and_watchdog():
    import datetime as dt
    from strokecopilot.compute import monitor
    state = {"created_utc": "2026-11-05T15:00:00+00:00", "dph": 0.5, "hours_limit": 5.5, "instance_id": 1}
    now = dt.datetime(2026, 11, 5, 16, 0, tzinfo=dt.timezone.utc)
    d = monitor.summarise("EXP-14", state, LOG, now=now)
    assert d["epochs_total"] == 300 and d["epochs_done"] == 2 and d["phase"] == "training"
    assert d["cost_so_far"] == 0.5 and d["exit_code"] is None
    assert d["watchdog_deadline_utc"].startswith("2026-11-05T20:44:26")
    assert d["cost_projected"] > d["cost_so_far"]
    assert "sub-stroke" not in json.dumps(d)
    d = monitor.summarise("EXP-14", {**state, "destroyed": True, "destroyed_utc": "2026-11-05T19:00:00+00:00"},
                          LOG.replace("#EXIT", "#EXIT 0"), now=now)
    assert d["phase"].startswith("finished") and d["exit_code"] == 0 and d["hours_up"] == 4.0


def test_monitor_reads_a_resumed_run_across_logs():
    from strokecopilot.compute import monitor
    first = ("2026-11-05 18:24:00.1: Epoch 269\n2026-11-05 18:24:01.1: train_loss -0.66\n2026-11-05 18:24:01.1: val_loss -0.18\n"
             "2026-11-05 18:24:01.1: Pseudo dice [np.float32(0.46)]\n2026-11-05 18:24:37.1: Epoch time: 37.6 s\n"
             "2026-11-05 18:25:52.9: Epoch 270\n")
    second = ("2026-11-05 19:33:32.9: Epoch 270\n2026-11-05 19:34:10.1: train_loss -0.67\n2026-11-05 19:34:10.1: val_loss -0.18\n"
              "2026-11-05 19:34:10.1: Pseudo dice [np.float32(0.51)]\n2026-11-05 19:34:17.6: Epoch time: 44.7 s\n")
    epochs = monitor.parse_log(first + second)
    assert [e["epoch"] for e in epochs] == [269, 270] and epochs[-1]["dice"] == 0.51
    assert "ls -tr" in monitor.REMOTE and "head -1" not in monitor.REMOTE
    assert "ls -tr" in remote.status("EXP-14") and "head -1" not in remote.status("EXP-14")


def test_up_refuses_an_offer_whose_contract_ends_too_soon(stubbed):
    prov, _, _ = stubbed
    prov._offers = [{"id": 11, "dph_total": 0.40, "end_date": time.time() + 3 * 3600}]
    with pytest.raises(pod.PodError, match="contract ends"):
        pod.up("EXP-14", 11, go=True, log=lambda m: None)
    assert prov.created == []
    assert round(vast.contract_hours({"end_date": 1000 + 7200}, now=1000), 1) == 2.0
    assert vast.contract_hours({"duration": 3600}) == 1.0
    assert "contract 2.0 h left" in Vast(key="k").describe({"id": 1, "end_date": time.time() + 7200 + 60})


def test_serve_runs_ask_for_a_48gb_gpu_with_parsas_policy():
    for exp in ("EXP-36", "EXP-37"):
        spec = jobs.get(exp)
        assert spec["filters"]["gpu_ram"] == {"gte": 44_000} and spec["filters"]["dph_total"] == {"lte": 0.90}
        assert "gpu_name" not in spec["filters"] and spec["disk_gb"] == 200
        assert spec["serve"] == {"models": ("gemma-3-27b", "medgemma-27b-text"), "allow_gated": True, "allow_medgemma": True}
    assert "rm -f /root/sc/hf_env" in remote.FORGET_TOKEN


def test_search_passes_a_runs_own_filters():
    class P:
        def __init__(self):
            self.calls = []

        def offers(self, filters=None, limit=10):
            self.calls.append(filters)
            return [{"id": 7, "dph_total": 0.6}]

    p = P()
    assert pod.search(p, rounds=2)[0]["id"] == 7 and p.calls == [None, None]  # the provider's own filters
    assert pod.find_offer(p, 7, rounds=1, filters=jobs.SERVE_FILTERS)["id"] == 7 and p.calls[-1] is jobs.SERVE_FILTERS
    assert pod.find_offer(p, 8, rounds=2, filters=jobs.SERVE_FILTERS) is None


def test_serve_scripts_and_tunnel():
    script = remote.start_serve(["vllm", "serve", "org/model", "--host", "127.0.0.1", "--port", "8000"])
    assert "tmux new-session -d -s serve" in script and "HF_HOME=/root/sc/hf" in script and "serve.log" in script
    assert "127.0.0.1" in script and "/root/sc/hf_env" in script
    env = remote.hf_env("hf_fake_not_a_real_token")
    assert "umask 077" in env and "export HF_TOKEN='hf_fake_not_a_real_token'" in env and "echo token stored" in env
    for bad in ("", "a'b", "a\nb"):
        with pytest.raises(ValueError):
            remote.hf_env(bad)
    assert "pip install -q vllm" in remote.SERVE_SETUP and "tmux kill-session -t serve" in remote.STOP_SERVE
    argv = pod.tunnel_argv({"host": "1.2.3.4", "port": 2222, "user": "root"}, 8000, 8000)
    assert argv[0] == "ssh" and argv[-1] == "root@1.2.3.4" and "-N" in argv
    assert argv[argv.index("-L") + 1] == "127.0.0.1:8000:127.0.0.1:8000"
