"""What do label-free and few-label corrections buy, using answers already saved by the benchmark runs?

raw     as saved
prior   label-free: divide each option's probability by its average over the (unlabelled) batch
fit     few labels: per-option bias (choice) or threshold (yes/no) fitted on half, scored on the other half,
        both directions averaged
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

logging.disable(logging.INFO)
from jev_benchmarking.cache import ResponseCache, request_key  # noqa: E402
from jev_benchmarking.tasks import get_tasks  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
TASKS = ["sst2", "ag_news", "banking77", "boolq", "clinc150", "toxic_chat", "emotion", "paws", "anli",
         "financial_phrasebank", "rotten_tomatoes", "sms_spam"]
F1_TASKS = {"toxic_chat", "sms_spam"}


def load(task, tag, examples):
    head = json.loads((HERE / "results/baselines/Qwen__Qwen3-4B" / f"{task}.json").read_text())["primary"]["head"]
    cache = ResponseCache(model=tag)
    P, y, options = [], [], None
    for e in examples:
        r = cache.get(request_key(tag, e.state, e.questions))
        if r is None:
            return None
        a = r["answers"][head]
        if a["type"] == "noul":
            P.append([1 - a["noul"], a["noul"]])
            y.append(int(bool(e.gold[head])))
        else:
            options = options or sorted(a["probabilities"])
            P.append([a["probabilities"][o] for o in options])
            y.append(options.index(e.gold[head]))
    return np.clip(np.array(P), 1e-6, 1), np.array(y)


def metric(task, pred, y):
    return f1_score(y, pred) if task in F1_TASKS else float((pred == y).mean())


def fit_bias(logp, y, steps=300, lr=0.5):
    b = np.zeros(logp.shape[1])
    onehot = np.eye(logp.shape[1])[y]
    for _ in range(steps):
        z = logp + b
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        b -= lr * (p - onehot).mean(0)
    return b


def fit_threshold(task, p, y):
    grid = np.linspace(0.02, 0.98, 49)
    return grid[int(np.argmax([metric(task, (p > t).astype(int), y) for t in grid]))]


def evaluate(task, P, y):
    logp = np.log(P)
    out = {"raw": metric(task, P.argmax(1), y)}
    out["prior"] = metric(task, (logp - np.log(P.mean(0))).argmax(1), y)
    idx = np.random.default_rng(0).permutation(len(y))
    halves = [idx[: len(y) // 2], idx[len(y) // 2:]]
    scores = []
    for a, b in (halves, halves[::-1]):
        if P.shape[1] == 2:
            t = fit_threshold(task, P[a, 1], y[a])
            pred = (P[b, 1] > t).astype(int)
        else:
            pred = (logp[b] + fit_bias(logp[a], y[a])).argmax(1)
        scores.append(metric(task, pred, y[b]))
    out["fit"] = float(np.mean(scores))
    return out


def main(tags):
    table = {}
    for name in TASKS:
        examples = get_tasks([name])[0].examples("eval", limit=300)
        for tag in tags:
            got = load(name, tag, examples)
            if got is not None:
                table.setdefault(tag, {})[name] = evaluate(name, *got)
    for tag in tags:
        if tag not in table:
            continue
        print(f"\n{tag}")
        print("  task".ljust(24) + "raw    prior  fit")
        for name, r in table[tag].items():
            print("  " + name.ljust(22) + "  ".join(f"{r[k]:.3f}" for k in ("raw", "prior", "fit")))
        m = {k: np.mean([r[k] for r in table[tag].values()]) for k in ("raw", "prior", "fit")}
        print("  mean".ljust(24) + "  ".join(f"{m[k]:.3f}" for k in m))
    (HERE / "results" / "offline_fixes.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main([f"hf:{m}" for m in sys.argv[1:]])
