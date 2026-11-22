"""EXP-67, the rulebook and test-case review: the analysis fixed in `docs/CLINICIAN_PACK.md` §6 before
any answer came back. Two reviewers, 69 scored items (A01-A47 the passages, B1a-B5b the fixtures' expected answers,
C01-C12 the rule conventions), each agree / disagree / unsure.

Reported: each reviewer's counts overall and per part; raw agreement on the three categories over items both
answered; Cohen's kappa with a 95% bootstrap interval over items (10,000 resamples, seed 20261104); PABAK beside it;
blanks counted as missing; and the items with any "disagree", with the comments. "Unsure" is a category, not missing.

Reviewers are R1 and R2 in file order. Names never leave the response file: the summary carries the self-reported role,
years band and how often they take part in acute stroke decisions, and each comment only as the consent allows
("do not quote me" drops the text). Expert opinion from two clinicians, never clinical validation.

    python -m strokecopilot.analysis.clinician_review <responses.csv> --out runs/EXP-67_clinician_review
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

CATEGORIES = ("agree", "disagree", "unsure")
PARTS = {"A": "rulebook passages", "B": "fixtures' expected answers", "C": "rule conventions"}
ITEM = re.compile(r"^([ABC]\d{1,2}[ab]?)\.\s*(.*)$", re.S)
N_ITEMS = {"A": 47, "B": 10, "C": 12}
N_BOOT, SEED = 10_000, 20261104


def category(answer):
    a = (answer or "").strip().lower()
    if not a:
        return None
    for c in CATEGORIES:
        if a.startswith(c):
            return c
    raise ValueError(f"Unexpected answer: {answer!r}")


def _find(headers, start):
    hits = [h for h in headers if h.strip().lower().startswith(start.lower())]
    if len(hits) != 1:
        raise ValueError(f"Expected one column starting {start!r}, found {len(hits)}.")
    return hits[0]


def read_responses(path):
    """[{role, years, frequency, quote, answers {item: category or None}, comments {item: text}, open {D1..D3: text}}],
    in file order, and {item: title}. The name column is never read."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    headers = list(rows[0]) if rows else []
    items = {}
    for h in headers:
        m = ITEM.match(h)
        if m and "comment" not in h:
            items[m.group(1)] = (h, m.group(2).strip())
    for part, n in N_ITEMS.items():
        got = sum(1 for i in items if i.startswith(part))
        if got != n:
            raise ValueError(f"Part {part}: {got} scored items in the file, the pack has {n}.")
    cols = {k: _find(headers, k) for k in ("P2.", "P3.", "P4.", "How may we quote you")}
    out = []
    for r in rows:
        quote = r[cols["How may we quote you"]].strip().lower()
        may_quote = not quote.startswith("do not quote")
        comments = {}
        for i in items:
            text = (r.get(f"{i} comment (optional)") or "").strip()
            if text:
                comments[i] = text if may_quote else "(not quoted, by the reviewer's choice)"
        out.append({"role": r[cols["P2."]].strip(), "years": r[cols["P3."]].strip(),
                    "frequency": r[cols["P4."]].strip().lower(), "quote": quote,
                    "answers": {i: category(r[h]) for i, (h, _) in items.items()}, "comments": comments,
                    "open": {d: ((r.get(_find(headers, d + ".")) or "").strip() if may_quote else "")
                             for d in ("D1", "D2", "D3")}})
    return out, {i: t for i, (_, t) in items.items()}


def counts(answers, part=None):
    sel = [a for i, a in answers.items() if part is None or i.startswith(part)]
    return {**{c: sum(a == c for a in sel) for c in CATEGORIES}, "blank": sum(a is None for a in sel), "items": len(sel)}


def kappa(a, b):
    """Cohen's kappa over paired category labels; None when chance agreement is 1 (both constant and equal)."""
    a, b = list(a), list(b)
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in CATEGORIES)
    return None if pe >= 1 else (po - pe) / (1 - pe)


def agreement(r1, r2, part=None, n_boot=N_BOOT, seed=SEED):
    """Raw agreement, kappa (bootstrap interval over items) and PABAK over the items both reviewers answered."""
    ids = sorted(i for i in r1 if (part is None or i.startswith(part)) and r1[i] is not None and r2.get(i) is not None)
    a, b = [r1[i] for i in ids], [r2[i] for i in ids]
    n = len(ids)
    if not n:
        return {"items_both_answered": 0}
    po = sum(x == y for x, y in zip(a, b)) / n
    out = {"items_both_answered": n, "agree_with_each_other": sum(x == y for x, y in zip(a, b)), "raw_agreement": po,
           "kappa": kappa(a, b), "pabak": (len(CATEGORIES) * po - 1) / (len(CATEGORIES) - 1),
           "both_agree": sum(x == y == "agree" for x, y in zip(a, b))}
    rng = np.random.default_rng(seed)
    boot = [kappa([a[j] for j in idx], [b[j] for j in idx]) for idx in rng.integers(0, n, size=(n_boot, n))]
    boot = [k for k in boot if k is not None]
    out["kappa_ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None
    return out


def summarise(reviewers, titles, n_boot=N_BOOT, seed=SEED):
    """The whole §6 summary for two reviewers. No name is in it."""
    if len(reviewers) != 2:
        raise ValueError(f"The plan is for two reviewers; the file has {len(reviewers)}.")
    r1, r2 = (r["answers"] for r in reviewers)
    out = {"experiment": "EXP-67", "instrument": 1, "scored_items": len(titles),
           "reviewers": [{"id": f"R{k + 1}", "role": r["role"], "years_in_practice": r["years"],
                          "takes_part_in_acute_stroke_decisions": r["frequency"], "quote": r["quote"],
                          "counts": {"all": counts(r["answers"]), **{p: counts(r["answers"], p) for p in PARTS}}}
                         for k, r in enumerate(reviewers)],
           "agreement": {"all": agreement(r1, r2, None, n_boot, seed),
                         **{p: agreement(r1, r2, p, n_boot, seed) for p in PARTS}}}
    out["items_with_a_disagree"] = [
        {"item": i, "title": titles[i][:160], "R1": r1[i], "R2": r2[i],
         "comments": {f"R{k + 1}": r["comments"][i] for k, r in enumerate(reviewers) if i in r["comments"]}}
        for i in sorted(titles) if "disagree" in (r1[i], r2[i])]
    out["comments"] = {f"R{k + 1}": r["comments"] for k, r in enumerate(reviewers)}
    out["open_answers"] = {f"R{k + 1}": r["open"] for k, r in enumerate(reviewers)}
    out["limits"] = ("Two reviewers; expert opinion, not clinical validation; roles self-reported; 'unsure' is a "
                     "category; the classification of disagreed items by two team members (pack §6) is not in this file.")
    return out


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="EXP-67: the agreement analysis fixed in the clinician pack")
    ap.add_argument("responses")
    ap.add_argument("--out", default="runs/EXP-67_clinician_review")
    args = ap.parse_args(argv)
    reviewers, titles = read_responses(args.responses)
    result = summarise(reviewers, titles)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics_instrument1.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({k: result[k] for k in ("reviewers", "agreement", "items_with_a_disagree")}, indent=1,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
