"""corpus/RULEBOOK.md (own words, DEC-18) -> corpus/passages.jsonl, with checks. Plan §5.3, §8.3.

The rulebook is written by the team. Each passage is a level-3 heading holding its ID, then `- key: value` lines, then
the text:

    ### AHA2026-4.6.1-within-4.5h
    - source: AHA2026
    - section: 4.6.1
    - pointer: Recommendation in §4.6.1; COR 1
    - page: e342
    - treatment: thrombolysis
    - must_have: yes
    - status: checked

    IV thrombolysis is recommended ...

The level-2 heading above a passage is its topic. Text before the first passage is a preamble and is ignored.

The copy check compares passages with the source documents' text on the local machine and reports passage IDs and
counts only. The PDFs are never sent to an LLM and their text is never printed.

    python -m strokecopilot.nlp.corpus.ingest build
    python -m strokecopilot.nlp.corpus.ingest copy-check AHA2026=/path/guideline.pdf DEFUSE3=/path/defuse3.pdf
"""

import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from strokecopilot.config import CORPUS_DIR

SOURCES = {
    "AHA2026": "2026 AHA/ASA guideline for the early management of acute ischaemic stroke (Stroke 2026;57:e316-e436)",
    "DAWN": "DAWN trial (Nogueira et al., N Engl J Med 2018;378:11-21)",
    "DEFUSE3": "DEFUSE-3 trial (Albers et al., N Engl J Med 2018;378:708-718; PMC author manuscript PMC6590673)",
}
TREATMENTS = ("thrombolysis", "thrombectomy", "both", "other")
STATUSES = ("draft", "checked", "reviewed")  # checked: compared with the PDF by a team member; reviewed: by a clinician
FIELDS = ("source", "section", "pointer", "page", "treatment", "must_have", "status")
MAX_WORDS = 120
COPY_RUN = 12  # this many consecutive words shared with a source is reported as possible copying

ID_RE = re.compile(r"^(AHA2026|DAWN|DEFUSE3)-[A-Za-z0-9.]+(-[a-z0-9.]+)+$")
_FIELD_RE = re.compile(r"^- ([a-z_]+):\s*(.*?)\s*$")

RULEBOOK = CORPUS_DIR / "RULEBOOK.md"
PASSAGES = CORPUS_DIR / "passages.jsonl"


@dataclass(frozen=True)
class Passage:
    id: str
    source: str
    section: str
    pointer: str
    page: str
    treatment: str
    must_have: bool
    status: str
    topic: str
    text: str

    @property
    def words(self):
        return len(self.text.split())


def _problems(fields, pid, line):
    out = []
    missing = [f for f in FIELDS if f not in fields]
    if missing:
        out.append(f"line {line}: {pid} lacks {', '.join(missing)}")
    extra = sorted(set(fields) - set(FIELDS))
    if extra:
        out.append(f"line {line}: {pid} has unknown fields {', '.join(extra)}")
    if not ID_RE.match(pid):
        out.append(f"line {line}: malformed ID {pid!r}")
    elif fields.get("source") and not pid.startswith(fields["source"] + "-"):
        out.append(f"line {line}: {pid} does not start with its source {fields['source']}")
    if fields.get("source") not in (None, *SOURCES):
        out.append(f"line {line}: {pid} has unknown source {fields['source']!r}")
    if fields.get("treatment") not in (None, *TREATMENTS):
        out.append(f"line {line}: {pid} has unknown treatment {fields['treatment']!r}")
    if fields.get("status") not in (None, *STATUSES):
        out.append(f"line {line}: {pid} has unknown status {fields['status']!r}")
    if fields.get("must_have") not in (None, "yes", "no"):
        out.append(f"line {line}: {pid} must_have is {fields['must_have']!r}, not yes or no")
    if fields.get("status") in ("checked", "reviewed") and not fields.get("page"):
        out.append(f"line {line}: {pid} is {fields['status']} but has no page")
    return out


def parse(text):
    """Passages in file order. Raises ValueError listing every problem."""
    passages, problems, seen = [], [], set()
    topic, current = "", None

    def close():
        if current is None:
            return
        pid, line, fields, body = current
        body_text = " ".join(" ".join(body).split())
        found = _problems(fields, pid, line)
        if not body_text:
            found.append(f"line {line}: {pid} has no text")
        if len(body_text.split()) > MAX_WORDS:
            found.append(f"line {line}: {pid} has {len(body_text.split())} words; the limit is {MAX_WORDS}")
        if pid in seen:
            found.append(f"line {line}: duplicate ID {pid}")
        seen.add(pid)
        problems.extend(found)
        if not found:
            passages.append(Passage(id=pid, topic=topic, text=body_text, must_have=fields["must_have"] == "yes",
                                    **{k: fields[k] for k in FIELDS if k != "must_have"}))

    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        if line.startswith("### "):
            close()
            current = (line[4:].strip(), n, {}, [])
        elif line.startswith("## ") or line.startswith("# "):
            close()
            current = None
            if line.startswith("## "):
                topic = line[3:].strip()
        elif current is not None:
            pid, _, fields, body = current
            match = _FIELD_RE.match(line)
            if match and not body:
                key, value = match.groups()
                if key in fields:
                    problems.append(f"line {n}: {pid} repeats {key}")
                fields[key] = value
            elif line.strip():
                body.append(line.strip())
    close()
    if problems:
        raise ValueError("Rulebook problems:\n" + "\n".join(problems))
    return passages


def read(path=RULEBOOK):
    return parse(Path(path).read_text(encoding="utf-8"))


def to_jsonl(passages):
    return "".join(json.dumps(asdict(p), ensure_ascii=False) + "\n" for p in passages)


def build(src=RULEBOOK, out=PASSAGES):
    """Write passages.jsonl from the rulebook; returns its sha256 (the corpus version)."""
    data = to_jsonl(read(src)).encode("utf-8")
    Path(out).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def version(path=PASSAGES):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path=PASSAGES):
    """{id: record} for retrieval and the verifier. The record keeps what a reader needs, not bookkeeping fields."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        out[rec["id"]] = {k: rec[k] for k in ("source", "section", "pointer", "treatment", "text")}
    return out


def must_have(path=PASSAGES):
    return [json.loads(line)["id"] for line in Path(path).read_text(encoding="utf-8").splitlines()
            if json.loads(line)["must_have"]]


def unchecked(passages):
    """IDs still in draft. Evaluation runs need this to be empty."""
    return [p.id for p in passages if p.status == "draft"]


# --- copy check ---------------------------------------------------------------------------------

def _words(text):
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def longest_shared_run(text, source_ngrams, n=COPY_RUN):
    """Length of the longest run of consecutive words of `text` that also appears in the source (0 if under n)."""
    words = _words(text)
    best = run = 0
    for i in range(len(words) - n + 1):
        if hash(tuple(words[i:i + n])) in source_ngrams:
            run += 1
            best = max(best, run + n - 1)
        else:
            run = 0
    return best


def ngrams(text, n=COPY_RUN):
    words = _words(text)
    return {hash(tuple(words[i:i + n])) for i in range(len(words) - n + 1)}


def copy_check(passages, source_texts, n=COPY_RUN):
    """{passage id: shared run length} for passages sharing n or more consecutive words with their own source."""
    grams = {src: ngrams(text, n) for src, text in source_texts.items()}
    flagged = {}
    for p in passages:
        if p.source in grams:
            run = longest_shared_run(p.text, grams[p.source], n)
            if run:
                flagged[p.id] = run
    return flagged


def pdf_text(path):
    from pypdf import PdfReader  # optional: pip install pypdf

    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["build"]:
        digest = build()
        passages = read()
        print(f"{len(passages)} passages, {len(unchecked(passages))} still draft; passages.jsonl sha256 {digest}")
    elif argv[:1] == ["copy-check"]:
        texts = {}
        for arg in argv[1:]:
            src, _, path = arg.partition("=")
            if src not in SOURCES:
                raise SystemExit(f"Unknown source {src!r}; use one of {', '.join(SOURCES)}")
            texts[src] = pdf_text(path)
        passages = read()
        flagged = copy_check(passages, texts)
        checked = sum(p.source in texts for p in passages)
        print(f"Compared {checked} passages with {', '.join(texts)}: {len(flagged)} share {COPY_RUN}+ consecutive words.")
        for pid, run in sorted(flagged.items()):
            print(f"  {pid}: {run} words")  # IDs and counts only, never the source text
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
