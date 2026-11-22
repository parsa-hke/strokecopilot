"""EXP-67: the agreement analysis of the clinician pack (§6), on a made-up response file."""

import csv

import pytest

from strokecopilot.analysis import clinician_review as cr

ITEMS = [f"A{i:02d}" for i in range(1, 48)] + [f"B{i}{s}" for i in range(1, 6) for s in "ab"] + \
        [f"C{i:02d}" for i in range(1, 13)]


def _file(tmp_path, answers1, answers2, quote2="By role only"):
    """A Google Forms export with two rows. answers: {item: text}; any item not given is 'Agree'."""
    headers = ["Timestamp", "Consent", "How may we quote you?", "May we thank you by name in the acknowledgements?",
               "P1. Your name (kept private unless you allow it above).", "P2. Your current role and specialty",
               "P3. Years in clinical practice since qualifying", "P4. How often do you take part?"]
    for i in ITEMS:
        headers += [f"{i}. some title for {i}", f"{i} comment (optional)"]
    headers += ["D1. Misleading?", "D2. A trap?", "D3. Anything else?"]
    rows = []
    for name, role, quote, answers in (("Secret One", "Stroke registrar", "By name and role", answers1),
                                       ("Secret Two", "Emergency physician", quote2, answers2)):
        row = ["t", "I agree", quote, "No", name, role, "5-10", "Regularly"]
        for i in ITEMS:
            row += [answers.get(i, "Agree"), "a note" if i == "A01" else ""]
        rows.append(row + ["none", "", ""])
    path = tmp_path / "responses.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    return path


def test_reads_69_items_and_never_the_name(tmp_path):
    reviewers, titles = cr.read_responses(_file(tmp_path, {}, {}))
    assert len(titles) == 69 and len(reviewers) == 2
    out = cr.summarise(reviewers, titles, n_boot=200)
    assert "Secret" not in str(out)
    assert out["reviewers"][0]["role"] == "Stroke registrar"
    assert out["reviewers"][0]["counts"]["all"] == {"agree": 69, "disagree": 0, "unsure": 0, "blank": 0, "items": 69}


def test_counts_agreement_pabak_and_blanks(tmp_path):
    a1 = {"A02": "Unsure / outside my practice", "A03": "Disagree", "B1a": "", "C01": "Disagree"}
    a2 = {"A02": "Unsure / outside my practice", "A03": "Agree", "C01": "Disagree", "C02": "Unsure / outside my practice"}
    reviewers, titles = cr.read_responses(_file(tmp_path, a1, a2))
    out = cr.summarise(reviewers, titles, n_boot=500)
    assert out["reviewers"][0]["counts"]["all"] == {"agree": 65, "disagree": 2, "unsure": 1, "blank": 1, "items": 69}
    assert out["reviewers"][0]["counts"]["B"]["blank"] == 1
    ag = out["agreement"]["all"]
    assert ag["items_both_answered"] == 68 and ag["agree_with_each_other"] == 66  # A03 and C02 differ; B1a is blank
    assert ag["raw_agreement"] == pytest.approx(66 / 68)
    assert ag["pabak"] == pytest.approx((3 * 66 / 68 - 1) / 2)
    assert ag["kappa_ci95"][0] <= ag["kappa"] <= ag["kappa_ci95"][1]
    assert [d["item"] for d in out["items_with_a_disagree"]] == ["A03", "C01"]
    assert out["items_with_a_disagree"][1]["R1"] == out["items_with_a_disagree"][1]["R2"] == "disagree"


def test_kappa_known_values():
    assert cr.kappa(["agree", "disagree"] * 5, ["agree", "disagree"] * 5) == pytest.approx(1.0)
    assert cr.kappa(["agree"] * 4, ["agree"] * 4) is None  # no variation: chance agreement is 1
    a = ["agree"] * 8 + ["unsure"] * 2
    b = ["agree"] * 6 + ["unsure"] * 4
    po, pe = 0.8, 0.8 * 0.6 + 0.2 * 0.4
    assert cr.kappa(a, b) == pytest.approx((po - pe) / (1 - pe))


def test_a_reviewer_who_declines_quotes_is_summarised_only(tmp_path):
    reviewers, titles = cr.read_responses(_file(tmp_path, {}, {}, quote2="Do not quote me; summarise my answers only"))
    out = cr.summarise(reviewers, titles, n_boot=200)
    assert out["comments"]["R1"] == {"A01": "a note"}
    assert out["comments"]["R2"] == {"A01": "(not quoted, by the reviewer's choice)"}
    assert out["open_answers"]["R2"]["D1"] == ""


def test_an_unexpected_answer_or_item_count_stops_the_run(tmp_path):
    with pytest.raises(ValueError):
        cr.read_responses(_file(tmp_path, {"A01": "Maybe"}, {}))
    with pytest.raises(ValueError):
        cr.category("perhaps")
