"""The notebooks as committed: none needs a GitHub token now that the repository is public, every install stays pinned
to REPO_REF, and notebook 62 keeps its stored outputs free of IDs, keys and local paths. Reads the files only."""

import json
import re
from pathlib import Path

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))
PIN = re.compile(r"git\+(https://github\.com/parsa-hke/strokecopilot\.git|\{repo\})@\{(REPO_REF|COMMIT)\}")
LEAKS = (r"/Users/", r"/content/", r"/var/folders", r"sub-stroke\d", r"sk-ant-", r"\bhf_[A-Za-z0-9]{20,}", r"\bID_[0-9a-f]{6,}")


def _code(notebook):
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def test_no_notebook_asks_for_a_github_token_and_installs_stay_pinned():
    assert len(NOTEBOOKS) >= 38
    installs = 0
    for path in NOTEBOOKS:
        text = path.read_text(encoding="utf-8")
        notebook = json.loads(text)
        assert notebook["nbformat"] == 4
        assert "GITHUB_TOKEN" not in text and "x-access-token" not in text, path.name
        for line in "\n".join(_code(notebook)).split("\n"):
            if "git+" in line:
                assert PIN.search(line), f"{path.name}: an install that is not pinned to REPO_REF: {line.strip()[:80]}"
                installs += 1
    assert installs >= 36  # every notebook that installs the package


def test_the_replay_notebook_keeps_clean_stored_outputs():
    path = next(p for p in NOTEBOOKS if p.name == "62_replay_no_drive.ipynb")
    text = path.read_text(encoding="utf-8")
    cells = [c for c in json.loads(text)["cells"] if c["cell_type"] == "code"]
    outputs = [o for c in cells for o in c["outputs"]]
    assert outputs and all(o["output_type"] == "stream" and o["name"] == "stdout" for o in outputs)  # no error, no warning
    printed = "".join("".join(o["text"]) for o in outputs)
    assert "OK: every cell matches its reported rates." in printed  # the replay's own last line
    assert '"case_id": "fixture-' in printed and re.search(r"^(ok  |FLAG) ", printed, re.M)  # the fixture and its claims
    assert re.search(r"^OK: the replay matched every reported rate\.", printed, re.M)
    assert not [p for p in LEAKS if re.search(p, text)]
    assert "userdata" not in text and "drive.mount" not in text  # no secret and no Drive
