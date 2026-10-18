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


def test_platt_fixes_an_overconfident_yes_bias():
    import random
    from anydecide.calibrate import Calibration, fit_yes_no
    rng = random.Random(0)
    labels = [rng.random() < 0.5 for _ in range(400)]
    # the model leans yes and is far too sure: margin = 6 for true cases, 3 for false ones, plus noise
    margins = [(6 if y else 3) + rng.gauss(0, 1.5) for y in labels]
    a, b = fit_yes_no(margins, [int(y) for y in labels])
    cal = Calibration("noul", a=a, b=b)
    import torch
    p_true = [cal.apply(torch.tensor([m / 2, -m / 2]))[0] for m, y in zip(margins, labels) if y]
    p_false = [cal.apply(torch.tensor([m / 2, -m / 2]))[0] for m, y in zip(margins, labels) if not y]
    assert sum(p_true) / len(p_true) > 0.7 and sum(p_false) / len(p_false) < 0.3


def test_choice_bias_moves_the_favoured_option(tmp_path):
    import torch
    from anydecide.calibrate import Calibration, fit_choice
    # the model always prefers option 0 by 2 nats, but the right answer is the one with the larger second score
    rows = [[0.0, -2.0 + d, -2.5] for d in (1.5, 2.5) * 50]
    gold = [0, 1] * 50
    t, bias = fit_choice(rows, gold)
    cal = Calibration("choice", temperature=t, bias=bias)
    assert max(range(3), key=lambda i: cal.apply(torch.tensor(rows[1]))[i]) == 1
    cal.save(tmp_path / "c.json")
    assert Calibration.load(tmp_path / "c.json") == cal
