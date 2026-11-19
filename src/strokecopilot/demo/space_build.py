"""Assemble the exact tree that is pushed to the Hugging Face Space (DEC-32 §2, §6), then scan it.

    python -m strokecopilot.demo.space_build --bank demo/space/build/bank --out demo/space/build/space

The tree: the Space's Dockerfile and README, the static page, the built bank, and `pkg/`, a copy of the package that
ships the schemas, fixtures, prompts and rulebook but **not the split** (`splits/folds.json` holds every case ID,
held-out ones included), no notebooks, no data and no run folders. The copied pyproject drops the split from the wheel.

The build refuses to finish if any file in the tree:
- holds an ISLES'24 case ID (all 149, development and held-out: compared in code, never printed);
- holds an RSNA ID, a key or token pattern, or an e-mail address;
- has a type outside the allowed list, or the tree contains a split file.
Messages carry counts and file names only. Pushing the tree to the Space is a publish, and only Parsa decides it.
"""

import argparse
import re
import shutil
from pathlib import Path

from strokecopilot.demo import bank

REPO = Path(__file__).resolve().parents[3]
SPACE_SRC = REPO / "demo" / "space"
PACKAGE_DIRS = ("schemas", "fixtures", "prompts", "corpus")
ALLOWED_SUFFIXES = {".py", ".json", ".jsonl", ".md", ".html", ".css", ".js", ".toml", ".txt", ".typed"}
ALLOWED_NAMES = {"Dockerfile", "LICENSE"}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
ALLOWED_EMAILS = {"noreply@anthropic.com"}


def _ignore(_dir, names):
    return [n for n in names if n in ("__pycache__", ".DS_Store") or n.endswith((".pyc", ".pyo"))]


def space_pyproject(text):
    """The repo's pyproject with the split taken out of the wheel."""
    out = [line for line in text.splitlines() if not line.strip().startswith('"splits"')]
    return "\n".join(out) + "\n"


def assemble(bank_dir, out_dir, repo=REPO):
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists; remove it or build into a new folder.")
    (out / "pkg").mkdir(parents=True)
    shutil.copy2(SPACE_SRC / "Dockerfile", out / "Dockerfile")
    shutil.copy2(SPACE_SRC / "README.md", out / "README.md")
    shutil.copytree(SPACE_SRC / "static", out / "static", ignore=_ignore)
    shutil.copytree(bank_dir, out / "bank", ignore=_ignore)
    shutil.copytree(Path(repo) / "src", out / "pkg" / "src", ignore=_ignore)
    for name in PACKAGE_DIRS:
        shutil.copytree(Path(repo) / name, out / "pkg" / name, ignore=_ignore)
    (out / "pkg" / "pyproject.toml").write_text(space_pyproject((Path(repo) / "pyproject.toml").read_text()))
    shutil.copy2(Path(repo) / "README.md", out / "pkg" / "README.md")
    return out


def check(tree, case_ids):
    """Problems in an assembled tree (counts and file names only). Empty means it may be pushed."""
    tree = Path(tree)
    problems = list(bank.scan(tree, case_ids))
    for path in sorted(tree.rglob("*")):
        if path.is_dir():
            if path.name == "splits":
                problems.append(f"{path.relative_to(tree)}: a split folder")
            continue
        rel = path.relative_to(tree)
        if path.name == "folds.json":
            problems.append(f"{rel}: a split file")
        if path.suffix not in ALLOWED_SUFFIXES and path.name not in ALLOWED_NAMES:
            problems.append(f"{rel}: file type not allowed")
            continue
        found = set(EMAIL.findall(path.read_text(errors="ignore"))) - ALLOWED_EMAILS
        if found:
            problems.append(f"{rel}: {len(found)} e-mail address(es)")
    return problems


def main(argv=None):
    from strokecopilot.data import splits

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bank", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    tree = assemble(args.bank, args.out)
    folds = splits.load_folds()
    ids = splits.development_ids(folds) + list(folds["heldout"])  # compared only; never printed
    problems = check(tree, ids)
    if problems:
        shutil.rmtree(tree)
        raise SystemExit(f"NOT OK: the Space tree failed the scan ({len(problems)} problems; tree removed): "
                         + "; ".join(problems[:10]))
    files = sum(1 for p in tree.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in tree.rglob("*") if p.is_file()) / 1e6
    print(f"OK: Space tree at {tree}: {files} files, {size:.1f} MB; scan clean (case IDs, RSNA IDs, keys, e-mail, "
          f"file types, no split).")


if __name__ == "__main__":
    main()
