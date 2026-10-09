"""Typed questions (choice / score / noul), rendered into one prompt whose answer is a single code token.
The prompt layout matches the open-model setup of the benchmark harness, so scores are directly comparable."""
import json
import re
import string
from dataclasses import dataclass

_ANON = re.compile(r"^(?:[A-Z]{1,2}|\d+)$")


@dataclass
class Rendered:
    text: str
    codes: list[str]   # what the model answers with
    keys: list[str]    # what the caller gets back
    kind: str


def fmt(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=1)


def codes_for(n: int) -> list[str]:
    letters = string.ascii_uppercase
    return list(letters[:n]) if n <= 26 else [a + b for a in letters for b in letters][:n]


def render(state, q: dict) -> Rendered:
    parts = ["Answer the question about the following state.", "State:\n" + fmt(state),
             "Question:\n" + fmt(q["instructions"])]
    kind = q["type"]
    if kind == "choice":
        keys = list(q["criteria"])
        codes = codes_for(len(keys))
        lines = []
        for code, key in zip(codes, keys):
            desc = q["criteria"][key]
            label = (fmt(desc) if desc is not None else key) if _ANON.match(key) else \
                (key if desc is None else f"{key} ({fmt(desc)})")
            lines.append(f"{code}: {label}")
        parts += ["Options:\n" + "\n".join(lines), "Reply with the code of the correct option only."]
    elif kind == "noul":
        codes, keys = ["Yes", "No"], ["yes", "no"]
        crit = q.get("criteria") or {}
        meaning = [f"{w} means: {fmt(crit[k])}" for w, k in (("Yes", "true"), ("No", "false")) if k in crit]
        if meaning:
            parts.append("\n".join(meaning))
        parts.append("Reply with Yes or No only.")
    elif kind == "score":
        levels = q["criteria"]
        if len(levels) > 10:
            raise ValueError("score questions support at most 10 levels")
        codes = keys = [str(i) for i in range(len(levels))]
        parts += ["Levels:\n" + "\n".join(f"{i}: {fmt(l)}" for i, l in enumerate(levels)),
                  "Reply with the number of the level only."]
    else:
        raise ValueError(f"unknown question type {kind!r}")
    return Rendered("\n\n".join(parts), codes, keys, kind)


def confidence(probs: list[float]) -> float:
    k = len(probs)
    return max(0.0, min(1.0, (k * max(probs) - 1) / (k - 1))) if k > 1 else 1.0


def answer(r: Rendered, probs: list[float], q: dict) -> dict:
    if r.kind == "noul":
        return {"type": "noul", "noul": probs[0]}
    dist = dict(zip(r.keys, probs))
    best = max(dist, key=dist.get)
    if r.kind == "choice":
        return {"type": "choice", "choice": best, "probabilities": dist, "confidence": confidence(probs)}
    return {"type": "score", "score": sum(i * p for i, p in enumerate(probs)),
            "legend": {str(i): l for i, l in enumerate(q["criteria"])}, "probabilities": dist,
            "confidence": confidence(probs)}
