"""Shell scripts that run on the pod. Kept as strings so tests can check them and `pod.py` can send them over SSH.

Layout on the pod (/root/sc): data/ (the tarball), repo/ (the package at the pinned commit), runs/EXP-<id>/ (outputs,
job.log, job.exit), watchdog.log.
"""

ROOT = "/root/sc"
# SSH sessions do not always inherit the image's PATH; the PyTorch images keep python in /opt/conda/bin.
PATH = "export PATH=/opt/conda/bin:/usr/local/bin:$PATH"

# Run first, before any copy: rsync and tmux are not in every image, and the PyTorch runtime image has no C compiler,
# which torch.compile (on by default in nnU-Net 2.8) needs (EXP-13, 5 Nov 2026: "Failed to find C compiler").
BOOTSTRAP = r"""set -e
mkdir -p /root/sc/data /root/sc/repo /root/sc/runs
if ! command -v rsync >/dev/null || ! command -v tmux >/dev/null || ! command -v gcc >/dev/null; then
  (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync tmux curl build-essential) >/dev/null 2>&1
fi
gcc --version | head -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nproc
free -g | awk '/Mem:/ {print $2 " GiB RAM"}'
df -BG /root | awk 'NR==2 {print $4 " free on /root"}'
"""

# Install the package from the archive sent up from the Mac, and nnU-Net from PyPI. No token is involved.
SETUP = r"""set -e
export PATH=/opt/conda/bin:/usr/local/bin:$PATH
cd /root/sc/repo
python -m pip install -q --upgrade pip >/dev/null
python -m pip install -q ".[isles]" "nnunetv2=={nnunet}" >/root/sc/pip.log 2>&1 || (tail -20 /root/sc/pip.log; exit 1)
python -c "import torch, strokecopilot; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"""

# The watchdog destroys the pod after the hour limit with the instance's own key, which Vast sets in every instance
# and which can only act on that instance. No key of ours is on the pod. If the destroy fails, it stops the container,
# which ends GPU billing (storage billing continues until `down`).
WATCHDOG = r"""#!/bin/bash
LIMIT_S="$1"
getvar() { local v="${!1}"; if [ -z "$v" ]; then v=$(tr '\0' '\n' </proc/1/environ | sed -n "s/^$1=//p"); fi; echo "$v"; }
echo "watchdog armed for ${LIMIT_S} s at $(date -u +%FT%TZ)" >> /root/sc/watchdog.log
sleep "$LIMIT_S"
ID=$(getvar CONTAINER_ID); KEY=$(getvar CONTAINER_API_KEY)
echo "hour limit reached at $(date -u +%FT%TZ); destroying instance ${ID}" >> /root/sc/watchdog.log
curl -s -o /dev/null -w "destroy HTTP %{http_code}\n" -X DELETE -H "Authorization: Bearer ${KEY}" \
  "https://console.vast.ai/api/v0/instances/${ID}/" >> /root/sc/watchdog.log 2>&1
sleep 120
echo "still alive after destroy; stopping the container" >> /root/sc/watchdog.log
kill -TERM 1
"""


def start_watchdog(limit_s):
    return (f"cat > /root/sc/watchdog.sh <<'WD'\n{WATCHDOG}WD\nchmod +x /root/sc/watchdog.sh\n"
            f"nohup setsid /root/sc/watchdog.sh {int(limit_s)} >/dev/null 2>&1 < /dev/null &\n"
            "sleep 1; pgrep -f watchdog.sh >/dev/null && echo watchdog running")


def start_job(exp, command):
    """Start the job detached under tmux; its exit code lands in job.exit when it ends."""
    run = f"/root/sc/runs/{exp}"
    inner = f"{PATH}; cd /root/sc && ( {command} ) > {run}/job.log 2>&1; echo $? > {run}/job.exit"
    return (f"mkdir -p {run} && rm -f {run}/job.exit && tmux new-session -d -s job {quote(inner)} "
            f"&& sleep 2 && tmux has-session -t job && echo job started")


def status(exp):
    run = f"/root/sc/runs/{exp}"
    return (f"if [ -f {run}/job.exit ]; then echo EXIT=$(cat {run}/job.exit); else echo RUNNING; fi; "
            f"tail -n 5 {run}/job.log 2>/dev/null; "
            # job.log lags (nnU-Net's output is block-buffered into it); nnU-Net's own log is current.
            # A resumed run writes a new log, so read them all, oldest first.
            f"t=$(ls -tr {run}/nnUNet_results/*/*/fold_*/training_log_*.txt 2>/dev/null); "
            f"[ -n \"$t\" ] && echo \"epochs done: $(cat $t | grep -c 'Epoch time')\" && cat $t | grep -E 'Epoch time|Pseudo dice' | tail -2; "
            f"tail -n 2 /root/sc/watchdog.log 2>/dev/null")


# A serve pod (jobs.SERVE_*): vLLM from PyPI, no package of ours, no data. HF_HOME keeps the weights on the pod's disk.
SERVE_SETUP = r"""set -e
export PATH=/opt/conda/bin:/usr/local/bin:$PATH
mkdir -p /root/sc/hf
python -m pip install -q --upgrade pip >/dev/null
python -m pip install -q vllm >/root/sc/pip.log 2>&1 || (tail -20 /root/sc/pip.log; exit 1)
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"""


def hf_env(token):
    """Write the Hugging Face token to a file only root can read, for gated weights (jobs.SERVE_POLICY allow_gated).
    The script travels on SSH's stdin, so the token is in no command line; it is never echoed or logged."""
    if not token or "\n" in token or "'" in token:
        raise ValueError("The Hugging Face token is missing or malformed.")
    return f"umask 077; cat > /root/sc/hf_env <<'HF'\nexport HF_TOKEN='{token}'\nHF\necho token stored"


def start_serve(argv):
    """Start `vllm serve` (open_models.serve_command) under tmux, replacing any server already running."""
    inner = (f"{PATH}; export HF_HOME=/root/sc/hf; [ -f /root/sc/hf_env ] && . /root/sc/hf_env; "
             f"exec {' '.join(quote(a) for a in argv)} > /root/sc/serve.log 2>&1")
    return (f"tmux kill-session -t serve 2>/dev/null; sleep 1; tmux new-session -d -s serve {quote(inner)} "
            f"&& sleep 2 && tmux has-session -t serve && echo serve started")


STOP_SERVE = ("tmux kill-session -t serve 2>/dev/null; sleep 8; "
              "nvidia-smi --query-gpu=memory.used --format=csv,noheader; echo serve stopped")
SERVE_LOG = "tail -n 25 /root/sc/serve.log 2>/dev/null; true"


def quote(s):
    return "'" + s.replace("'", "'\"'\"'") + "'"
