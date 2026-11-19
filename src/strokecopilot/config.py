"""Paths, constants, Drive layout and slim-format version pins (plan §4.2, §5.4)."""

from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent.parent


def _resource_dir(name: str) -> Path:
    # A wheel install ships schemas/ and fixtures/ inside the package (see pyproject.toml);
    # an editable install or a source checkout reads them from the repo root.
    packaged = _PKG_DIR / name
    return packaged if packaged.is_dir() else _REPO_ROOT / name


SCHEMAS_DIR = _resource_dir("schemas")
FIXTURES_DIR = _resource_dir("fixtures")
FIXTURE_NOTES_DIR = FIXTURES_DIR / "notes"
SPLITS_DIR = _resource_dir("splits")
PROMPTS_DIR = _resource_dir("prompts")
CORPUS_DIR = _resource_dir("corpus")  # own-words rulebook and passages.jsonl (DEC-18)

# Contract (plan §6)
VISION_SCHEMA_VERSION = "0.1"

# Google Drive layout (plan §4.2)
DRIVE_ROOT = Path("/content/drive/MyDrive/StrokeCopilot")
DRIVE_RAW = DRIVE_ROOT / "raw"
DRIVE_SLIM = DRIVE_ROOT / "slim"
DRIVE_RUNS = DRIVE_ROOT / "runs"
DRIVE_ARTIFACTS = DRIVE_ROOT / "artifacts"



def local_drive_root():
    """The StrokeCopilot folder of Google Drive for desktop on this computer: STROKECOPILOT_DRIVE if set, else the one
    GoogleDrive-<account> folder that holds it (as compute/pod.py finds it), so no account name lives in the code.
    None when there is no such folder or more than one."""
    import os

    env = os.environ.get("STROKECOPILOT_DRIVE")
    if env:
        return Path(env).expanduser()
    hits = sorted(Path.home().glob("Library/CloudStorage/GoogleDrive-*/My Drive/StrokeCopilot"))
    return hits[0] if len(hits) == 1 else None


# Colab local disk, where slim tarballs are extracted (plan §4.3)
LOCAL_ROOT = Path("/content")

# Slim tarball versions (plan §5.4). Immutable: a preprocessing change means a new version.
ISLES24_SLIM = "isles24_slim_v1.tar"
RSNA_SLIM = "rsna_slim_v1.tar"
