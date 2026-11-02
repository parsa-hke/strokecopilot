"""Summarise a 7z archive from its file list, without extracting anything (plan §7, Phase 1).

Files are grouped into patterns by replacing the case ID in their path, so a layout of thousands of
files reads as a short table. In a solid archive one file cannot be pulled out alone: 7z decodes its
block from the start up to that file. `decode_bytes` is that cost for a whole pattern, which is what
decides whether single modalities can be extracted cheaply.
"""

import json
import os
import re
import subprocess
import time
from collections import defaultdict

CASE_RE = re.compile(r"sub-[A-Za-z]*\d+")
PERFUSION_MAPS = ("cbf", "cbv", "mtt", "tmax")
SECTION_RULE = "----------"
MAX_PATTERNS_SHOWN = 60
MAX_SHARED_SHOWN = 40
MAX_MISSING_SHOWN = 20


def _run_with_heartbeat(cmd, stdout, timeout_s, heartbeat_s, doing):
    """Run a 7z command, printing progress, and return the seconds it took."""
    started = last = time.time()
    proc = subprocess.Popen(cmd, stdout=stdout, stderr=subprocess.PIPE, text=True)
    while proc.poll() is None:
        time.sleep(min(heartbeat_s, 1))
        now = time.time()
        if now - started > timeout_s:
            proc.kill()
            raise TimeoutError(f"7z took over {timeout_s} s {doing}. Reconnect and run again.")
        if now - last >= heartbeat_s:
            print(f"Still {doing}: {now - started:.0f} s", flush=True)
            last = now
    if proc.returncode:
        raise RuntimeError(f"7z exited with {proc.returncode}: {proc.stderr.read()[-800:]}")
    return time.time() - started


def list_archive(archive, out_path, seven_zip="7z", timeout_s=1800, heartbeat_s=30):
    """Write `7z l -slt` output to out_path. Reads the archive only; out_path appears only on success."""
    part = f"{out_path}.part"
    with open(part, "w") as out:
        seconds = _run_with_heartbeat([seven_zip, "l", "-slt", archive], out, timeout_s, heartbeat_s, "reading the file list")
    os.replace(part, out_path)
    return seconds


def extract_members(archive, dest, members, exclude=(), seven_zip="7z", timeout_s=1800, heartbeat_s=30):
    """Extract paths or wildcards from the archive into dest, keeping their folders."""
    cmd = [seven_zip, "x", archive, f"-o{dest}", "-y", "-bd", *members, *(f"-xr!{pattern}" for pattern in exclude)]
    return _run_with_heartbeat(cmd, subprocess.DEVNULL, timeout_s, heartbeat_s, "extracting")


def copy_with_progress(src, dst, every_s=60, chunk_bytes=16 * 1024 ** 2):
    """Copy a large file, printing progress, and return the seconds it took. Checks the final size."""
    total = os.path.getsize(src)
    started = last = time.time()
    done = 0
    with open(src, "rb") as reader, open(f"{dst}.part", "wb") as writer:
        while chunk := reader.read(chunk_bytes):
            writer.write(chunk)
            done += len(chunk)
            now = time.time()
            if now - last >= every_s:
                print(f"Copied {done / 1024 ** 3:.1f} of {total / 1024 ** 3:.1f} GiB, {done / 1024 ** 2 / (now - started):.0f} MiB/s", flush=True)
                last = now
    if os.path.getsize(f"{dst}.part") != total:
        raise OSError(f"Copy of {src} is the wrong size.")
    os.replace(f"{dst}.part", dst)
    return time.time() - started


def _fields(lines):
    fields = {}
    for line in lines:
        key, sep, value = line.partition(" =")
        if sep:
            fields[key] = value.strip()
    return fields


def parse_slt(text):
    """Parse `7z l -slt` output into (archive properties, entries in stored order)."""
    lines = text.splitlines()
    if "--" not in lines or SECTION_RULE not in lines:
        raise ValueError("Not a 7z -slt listing: section markers are missing.")
    start, rule = lines.index("--"), lines.index(SECTION_RULE)
    archive = _fields(lines[start + 1 : rule])

    entries = []
    for chunk in "\n".join(lines[rule + 1 :]).split("\n\n"):
        fields = _fields(chunk.splitlines())
        if "Path" not in fields:
            continue
        attributes = fields.get("Attributes", "")
        entries.append(
            {
                "path": fields["Path"],
                "size": int(fields.get("Size") or 0),
                "block": fields.get("Block") or None,
                "encrypted": fields.get("Encrypted") == "+",
                "is_dir": fields.get("Folder") == "+" or attributes.startswith("D"),
            }
        )
    return archive, entries


def _extension(name):
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return os.path.splitext(name)[1] or "(none)"


def decode_bytes(files):
    """Bytes 7z must decode to extract `files`: each block from its start up to the last wanted file."""
    return sum(f["decode_to"] for f in {f["block"] or f["path"]: f for f in sorted(files, key=lambda f: f["decode_to"])}.values())


def _with_decode_offsets(entries):
    position = defaultdict(int)
    files = []
    for entry in entries:
        if entry["is_dir"]:
            continue
        key = entry["block"] or entry["path"]
        position[key] += entry["size"]
        files.append({**entry, "decode_to": position[key]})
    return files


def summarise(archive, entries):
    files = _with_decode_offsets(entries)
    cases = sorted({m.group() for f in files if (m := CASE_RE.search(f["path"]))})

    top_level = defaultdict(lambda: {"files": 0, "bytes": 0})
    extensions = defaultdict(lambda: {"files": 0, "bytes": 0})
    by_pattern = defaultdict(list)
    shared = []
    for f in files:
        for bucket in (top_level[f["path"].split("/")[0]], extensions[_extension(f["path"].rsplit("/", 1)[-1])]):
            bucket["files"] += 1
            bucket["bytes"] += f["size"]
        if CASE_RE.search(f["path"]):
            by_pattern[CASE_RE.sub("{case}", f["path"])].append(f)
        else:
            shared.append({"path": f["path"], "bytes": f["size"], "decode_bytes": f["decode_to"]})

    patterns = []
    for pattern, group in sorted(by_pattern.items()):
        present = {CASE_RE.search(f["path"]).group() for f in group}
        patterns.append(
            {
                "pattern": pattern,
                "files": len(group),
                "cases": len(present),
                "bytes": sum(f["size"] for f in group),
                "decode_bytes": decode_bytes(group),
                "missing_cases": sorted(set(cases) - present),
            }
        )

    perfusion_maps = {}
    for name in PERFUSION_MAPS:
        word = re.compile(rf"(?<![a-z]){name}(?![a-z])", re.IGNORECASE)
        hits = [p for p in patterns if word.search(p["pattern"].rsplit("/", 1)[-1])]
        perfusion_maps[name] = {"max_cases": max((p["cases"] for p in hits), default=0), "patterns": [p["pattern"] for p in hits]}

    return {
        "archive": {
            "type": archive.get("Type"),
            "physical_size": int(archive.get("Physical Size") or 0),
            "headers_size": int(archive.get("Headers Size") or 0),
            "method": archive.get("Method"),
            "solid": archive.get("Solid") == "+",
            "blocks": int(archive.get("Blocks") or 0),
        },
        "n_files": len(files),
        "n_dirs": len(entries) - len(files),
        "unpacked_bytes": sum(f["size"] for f in files),
        "encrypted_files": sum(f["encrypted"] for f in files),
        "n_cases": len(cases),
        "first_case": cases[0] if cases else None,
        "last_case": cases[-1] if cases else None,
        "top_level": dict(top_level),
        "extensions": dict(extensions),
        "perfusion_maps": perfusion_maps,
        "shared_files": shared,
        "patterns": patterns,
    }


def find_problems(summary, expected_size, expected_cases):
    """Things that disagree with what we expected. These are findings to report, not crashes."""
    problems = []
    if summary["n_files"] == 0:
        problems.append("The listing has no files.")
    if summary["archive"]["physical_size"] != expected_size:
        problems.append(f"Archive size {summary['archive']['physical_size']} is not the verified {expected_size}.")
    if summary["n_cases"] != expected_cases:
        problems.append(f"Found {summary['n_cases']} case IDs, expected {expected_cases}.")
    if summary["encrypted_files"]:
        problems.append(f"{summary['encrypted_files']} files are encrypted.")
    for name, found in summary["perfusion_maps"].items():
        if found["max_cases"] < summary["n_cases"]:
            problems.append(f"{name} map: best pattern covers {found['max_cases']} of {summary['n_cases']} cases.")
    return problems


def _gib(n):
    return f"{n / 1024 ** 3:.3f}"


def format_report(summary, problems, local_free_bytes=None):
    a = summary["archive"]
    out = [
        f"ARCHIVE  type={a['type']} method={a['method']} solid={a['solid']} blocks={a['blocks']} "
        f"size={_gib(a['physical_size'])} GiB headers={a['headers_size']} bytes",
        f"CONTENTS files={summary['n_files']} dirs={summary['n_dirs']} unpacked={_gib(summary['unpacked_bytes'])} GiB",
        f"CASES    {summary['n_cases']} ({summary['first_case']} to {summary['last_case']})",
    ]
    if local_free_bytes is not None:
        fits = "fits" if summary["unpacked_bytes"] < local_free_bytes else "DOES NOT FIT"
        out.append(f"DISK     local free={_gib(local_free_bytes)} GiB: a full unpack {fits}")

    out += ["", f"PROBLEMS ({len(problems)})"] + [f"  {p}" for p in problems]

    out += ["", "PERFUSION MAPS"]
    for name, found in summary["perfusion_maps"].items():
        out.append(f"  {name}: up to {found['max_cases']} cases in {found['patterns'] or 'no pattern'}")

    out += ["", "TOP LEVEL (files, GiB)"]
    out += [f"  {k}: {v['files']}, {_gib(v['bytes'])}" for k, v in sorted(summary["top_level"].items())]
    out += ["", "EXTENSIONS (files, GiB)"]
    out += [f"  {k}: {v['files']}, {_gib(v['bytes'])}" for k, v in sorted(summary["extensions"].items())]

    shared = summary["shared_files"]
    out += ["", f"FILES NOT TIED TO A CASE ({len(shared)}): GiB, GiB to decode, path"]
    out += [f"  {_gib(f['bytes'])}  {_gib(f['decode_bytes'])}  {f['path']}" for f in shared[:MAX_SHARED_SHOWN]]
    if len(shared) > MAX_SHARED_SHOWN:
        out.append(f"  ... {len(shared) - MAX_SHARED_SHOWN} more in the summary JSON")

    patterns = summary["patterns"]
    out += ["", f"PATTERNS ({len(patterns)}): cases, files, GiB, GiB to decode, pattern"]
    for p in patterns[:MAX_PATTERNS_SHOWN]:
        out.append(f"  {p['cases']:>4} {p['files']:>6} {_gib(p['bytes']):>9} {_gib(p['decode_bytes']):>9}  {p['pattern']}")
        missing = p["missing_cases"]
        if missing:
            shown = ", ".join(missing[:MAX_MISSING_SHOWN])
            more = f" and {len(missing) - MAX_MISSING_SHOWN} more" if len(missing) > MAX_MISSING_SHOWN else ""
            out.append(f"       missing: {shown}{more}")
    if len(patterns) > MAX_PATTERNS_SHOWN:
        out.append(f"  ... {len(patterns) - MAX_PATTERNS_SHOWN} more in the summary JSON")
    return "\n".join(out)


def write_summary(summary, path):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
