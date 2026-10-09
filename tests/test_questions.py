import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anydecide.model import fit_temperature  # noqa: E402
from anydecide.questions import answer, confidence, render  # noqa: E402

QS = {
    "dept": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "payments", "tech": None}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
    "churn": {"type": "noul", "instructions": "Does the user threaten to leave?"},
    "mc": {"type": "choice", "instructions": "Pick", "criteria": {"A": "red", "B": "blue"}},
}


def test_render_formats():
    r = render("Refund me or I cancel.", QS["dept"])
    assert r.codes == ["A", "B"] and r.keys == ["billing", "tech"]
    assert "A: billing (payments)\nB: tech" in r.text and r.text.endswith("Reply with the code of the correct option only.")
    assert render("x", QS["churn"]).codes == ["Yes", "No"]
    assert render("x", QS["urgency"]).codes == ["0", "1", "2"]
    assert "A: red\nB: blue" in render("x", QS["mc"]).text   # anonymous keys show only their text


def test_matches_the_benchmark_prompts():
    ref = Path(__file__).resolve().parent.parent / "ref" / "jev-benchmarking"
    if not ref.exists():
        pytest.skip("jev-benchmarking not cloned")
    sys.path.insert(0, str(ref))
    from jev_benchmarking.openmodel.prompts import render as bench_render
    for q in QS.values():
        for state in ("plain text", {"premise": "a", "hypothesis": "b"}):
            assert render(state, q).text == bench_render(state, q).text


def test_answers_and_confidence():
    r = render("x", QS["urgency"])
    a = answer(r, [0.7, 0.2, 0.1], QS["urgency"])
    assert abs(a["score"] - 0.4) < 1e-9 and a["legend"]["2"] == "high"
    assert confidence([0.5, 0.5]) == 0.0 and confidence([1.0, 0.0]) == 1.0
    assert answer(render("x", QS["churn"]), [0.9, 0.1], QS["churn"]) == {"type": "noul", "noul": 0.9}


def test_fit_temperature_softens_overconfident_scores():
    # always 99% sure, right only half the time: the best temperature flattens the distribution
    rows = [[0.0, -4.6]] * 50 + [[-4.6, 0.0]] * 50
    gold = [0] * 25 + [1] * 25 + [1] * 25 + [0] * 25
    assert fit_temperature(rows, gold) > 3
