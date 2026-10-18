"""Compare ways of turning a model's Yes/No margin into a calibrated probability.

Reads results/noul/<model>/<task>.jsonl from noul_collect.py and writes results/noul/summary.json.
Methods:
  raw         sigmoid(z)
  comp_opp    average with the natural opposite question: (p + 1 - p_opp) / 2
  comp_neg    average with the plain negation ("is it false that ...")
  temp        sigmoid(z / T), T fitted on n labels
  platt       sigmoid(a z + b), fitted on n labels
  platt2      logistic on [z, z_opp], fitted on n labels (uses the opposite question as a second feature)
  isotonic    monotone fit of y on z (pool adjacent violators), fitted on n labels
Each fitted method is run for several label budgets n, over several random fit/test splits.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "results" / "noul"
BUDGETS = [25, 50, 100, 200, 400]
SPLITS = 10
TEST = 500


def sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def ece(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def auroc(p, y):
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")


def metrics(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    pred = p >= 0.5
    tp, fp, fn = (pred & (y == 1)).sum(), (pred & (y == 0)).sum(), (~pred & (y == 1)).sum()
    return {"ece": ece(p, y), "brier": float(np.mean((p - y) ** 2)), "nll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "auroc": auroc(p, y), "acc": float(np.mean(pred == y)), "f1": float(2 * tp / max(1, 2 * tp + fp + fn))}


def fit_logistic(X, y, l2=1e-2, iters=50):
    X = np.column_stack([X, np.ones(len(X))])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sig(X @ w)
        g = X.T @ (p - y) + l2 * np.r_[w[:-1], 0]
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return lambda Z: sig(np.column_stack([Z, np.ones(len(Z))]) @ w)


def fit_temp(z, y):
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 200))
    nll = [-np.mean(y * np.log(np.clip(sig(z / t), 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - sig(z / t), 1e-9, 1))) for t in grid]
    t = grid[int(np.argmin(nll))]
    return lambda Z: sig(Z / t)


def fit_isotonic(z, y):
    order = np.argsort(z)
    xs, ys = z[order], y[order].astype(float)
    blocks = [[v, 1.0, x, x] for v, x in zip(ys, xs)]   # mean, weight, lo, hi
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            m2, w2, lo2, _ = out.pop(-2)
            m1, w1, _, hi1 = out.pop()
            out.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, lo2, hi1])
    knots = np.array([(b[2] + b[3]) / 2 for b in out])
    vals = np.clip(np.array([b[0] for b in out]), 0.01, 0.99)
    return lambda Z: np.interp(Z, knots, vals)


def run_task(rows):
    y = np.array([r["y"] for r in rows])
    z, zo, zn = (np.array([r[k] for r in rows]) for k in ("z", "z_opp", "z_neg"))
    res = {"n": len(rows), "base_rate": float(y.mean()),
           "complement_gap_opp": float(np.mean(np.abs(sig(z) + sig(zo) - 1))),
           "complement_gap_neg": float(np.mean(np.abs(sig(z) + sig(zn) - 1)))}
    fixed = {"raw": sig(z), "comp_opp": (sig(z) + 1 - sig(zo)) / 2, "comp_neg": (sig(z) + 1 - sig(zn)) / 2}
    rng = np.random.default_rng(0)
    acc = {}
    for s in range(SPLITS):
        perm = rng.permutation(len(rows))
        test, pool = perm[:TEST], perm[TEST:]
        for name, p in fixed.items():
            acc.setdefault((name, 0), []).append(metrics(p[test], y[test]))
        for n in BUDGETS:
            if n > len(pool):
                continue
            fit = pool[:n]
            if len(set(y[fit])) < 2:
                continue
            fits = {"temp": fit_temp(z[fit], y[fit])(z[test]),
                    "platt": fit_logistic(z[fit][:, None], y[fit])(z[test][:, None]),
                    "platt2": fit_logistic(np.column_stack([z[fit], zo[fit]]), y[fit])(np.column_stack([z[test], zo[test]])),
                    "isotonic": fit_isotonic(z[fit], y[fit])(z[test])}
            for name, p in fits.items():
                acc.setdefault((name, n), []).append(metrics(p, y[test]))
    res["methods"] = {f"{m}@{n}" if n else m: {k: float(np.nanmean([d[k] for d in v])) for k in v[0]} for (m, n), v in acc.items()}
    return res


def main():
    summary = {}
    for model_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        for f in sorted(model_dir.glob("*.jsonl")):
            rows = [json.loads(l) for l in f.open(encoding="utf-8")]
            if len(rows) < TEST + 50:
                continue
            summary.setdefault(model_dir.name, {})[f.stem] = run_task(rows)
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=1))
    for model, tasks in summary.items():
        print(f"\n{model}")
        print(f"{'task':16} {'raw':>7} {'comp_opp':>9} {'comp_neg':>9} {'temp@100':>9} {'platt@100':>10} {'platt2@100':>11} {'iso@100':>8}   (ECE)")
        for task, r in tasks.items():
            m = r["methods"]
            cell = lambda k: f"{m[k]['ece']:.3f}" if k in m else "  -  "
            print(f"{task:16} {cell('raw'):>7} {cell('comp_opp'):>9} {cell('comp_neg'):>9} {cell('temp@100'):>9} {cell('platt@100'):>10} {cell('platt2@100'):>11} {cell('isotonic@100'):>8}")


if __name__ == "__main__":
    main()
