"""Refuse to start a pod if a secret is in the repo tree (DEC-21, plan §4.4).

Checks every file git would track or could add (tracked, plus untracked files that are not ignored) for the exact
values of the API keys in this shell's environment, and for the shapes of common keys and private keys. Reports file
names only, never the matching text.
"""

import os
import re
import subprocess
from pathlib import Path

# Every key name the code may read through providers.secret() or a platform client. VLLM_API_KEY is VLLMJSON's
# conventional key_name; OPENAI_API_KEY is what the openai client falls back to; HUGGING_FACE_HUB_TOKEN is HF_TOKEN's
# older name.
ENV_KEYS = ("VAST_API_KEY", "RUNPOD_API_KEY", "ANTHROPIC_API_KEY", "GITHUB_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
            "KAGGLE_KEY", "VLLM_API_KEY", "OPENAI_API_KEY")
PATTERNS = {
    "Anthropic key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "GitHub token": re.compile(r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "AWS key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"),
}
MAX_BYTES = 5_000_000


class SecretInRepo(RuntimeError):
    pass


def repo_files(root):
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=root, capture_output=True, text=True, check=True).stdout
    return [Path(root) / name for name in out.split("\0") if name]


def scan(root, environ=None):
    """[(file, what)] for every secret found; empty when clean."""
    environ = os.environ if environ is None else environ
    values = {name: environ[name] for name in ENV_KEYS if len(environ.get(name, "")) >= 12}
    found = []
    for path in repo_files(root):
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
            text = path.read_text(errors="ignore")
        except (OSError, UnicodeError):
            continue
        for name, value in values.items():
            if value in text:
                found.append((str(path.relative_to(root)), f"the value of ${name}"))
        for what, pattern in PATTERNS.items():
            if pattern.search(text):
                found.append((str(path.relative_to(root)), what))
    return found


def require_clean(root, environ=None):
    found = scan(root, environ)
    if found:
        listing = "; ".join(f"{f} ({what})" for f, what in found)
        raise SecretInRepo(f"Refusing to start: a secret is in the repo tree: {listing}. Remove it first.")
