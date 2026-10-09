"""Training-free ways to raise accuracy, measured on the benchmark tasks.

zero      the benchmark's own prompt (state, question, lettered options)
layout    question and options first, state last (control for the few-shot layout)
shot4/8   layout + k labelled examples from the task's dev split
perm3     choice only: average over three random option orders
stage2    choice with many options only: first pass picks the top 5, second pass decides among them
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

logging.disable(logging.INFO)
from jev_benchmarking.openmodel.prompts import render  # noqa: E402
from jev_benchmarking.openmodel.scorer import HFScorer  # noqa: E402
from jev_benchmarking.tasks import get_tasks  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
TASKS = ["sst2", "ag_news", "banking77", "boolq", "clinc150", "toxic_chat", "emotion", "paws", "anli",
         "financial_phrasebank", "rotten_tomatoes", "sms_spam"]
F1_TASKS = {"toxic_chat", "sms_spam"}


def fmt(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=1)


def gold_code(rq, gold):
    if rq.kind == "noul":
        return "Yes" if gold else "No"
    return rq.codes[rq.keys.index(str(gold))]


def split_prompt(rq_text, state):
    """The benchmark prompt is 'intro, State, Question..., reply line'. Pull the question part out."""
    marker = "State:\n" + fmt(state)
    head, tail = rq_text.split(marker, 1)
    return tail.strip()


def few_shot(state, q, head, shots, single):
    rq = render(state, q, single)
    parts = ["Answer the question about a state.", split_prompt(rq.text, state)]
    for s_state, s_q, s_gold in shots:
        parts.append("State:\n" + fmt(s_state) + "\nAnswer: " + gold_code(render(s_state, s_q, single), s_gold))
    parts.append("State:\n" + fmt(state) + "\nAnswer:")
    return "\n\n".join(parts), rq


def probs(scorer, texts, codes):
    res = scorer.score([scorer.prompt_ids(t) for t in texts], codes)
    out = []
    for r in res:
        lp = np.array(r.logprobs)
        p = np.exp(lp - lp.max())
        out.append(p / p.sum())
    return out


def metric(task, pred, y):
    pred, y = np.array(pred), np.array(y)
    return float(f1_score(y, pred)) if task in F1_TASKS else float((pred == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--vram", type=float, default=0.8)
    args = ap.parse_args()

    torch.cuda.set_per_process_memory_fraction(args.vram)
    scorer = HFScorer(args.model, max_batch_tokens=4096)
    single = scorer.is_single_token
    out_path = HERE / "results" / f"prompt_methods_{args.model.split('/')[-1]}.json"
    table = json.loads(out_path.read_text()) if out_path.exists() else {}

    for name in args.tasks:
        if name in table:
            continue
        task = get_tasks([name])[0]
        head = "toxic" if name == "toxic_chat" else "answer"
        ex = task.examples("eval", limit=args.limit)
        dev = task.examples("dev", limit=8) if task.splits.get("dev") else []
        shots = [(d.state, d.questions[head], d.gold[head]) for d in dev]
        rqs = [render(e.state, e.questions[head], single) for e in ex]
        kind = rqs[0].kind
        y = [int(bool(e.gold[head])) if kind == "noul" else rq.keys.index(str(e.gold[head])) for e, rq in zip(ex, rqs)]
        res = {"n": len(ex), "options": len(rqs[0].codes), "dev_shots": len(shots)}

        def run(texts, code_lists):
            ps = probs(scorer, texts, code_lists)
            return [int(p[0] > 0.5) if kind == "noul" else int(np.argmax(p)) for p in ps], ps

        pred, p_zero = run([rq.text for rq in rqs], [rq.codes for rq in rqs])
        res["zero"] = metric(name, pred, y)

        for label, k in (("layout", 0), ("shot4", 4), ("shot8", 8)):
            if k > len(shots):
                continue
            texts = [few_shot(e.state, e.questions[head], head, shots[:k], single)[0] for e in ex]
            pred, _ = run(texts, [rq.codes for rq in rqs])
            res[label] = metric(name, pred, y)

        if kind == "choice":
            rng = random.Random(0)
            acc = [np.array(p) for p in p_zero]
            for _ in range(2):
                texts, perms = [], []
                for e in ex:
                    q = dict(e.questions[head])
                    keys = list(q["criteria"])
                    rng.shuffle(keys)
                    q["criteria"] = {k: q["criteria"][k] for k in keys}
                    perms.append(keys)
                    texts.append(render(e.state, q, single).text)
                _, ps = run(texts, [rq.codes for rq in rqs])
                for i, (p, keys) in enumerate(zip(ps, perms)):
                    back = [keys.index(k) for k in rqs[i].keys]
                    acc[i] = acc[i] + np.array(p)[back]
            res["perm3"] = metric(name, [int(np.argmax(a)) for a in acc], y)

            if len(rqs[0].codes) > 26:
                texts, tops = [], []
                for e, rq, p in zip(ex, rqs, p_zero):
                    top = [rq.keys[j] for j in np.argsort(p)[::-1][:5]]
                    q = dict(e.questions[head])
                    q["criteria"] = {k: q["criteria"][k] for k in top}
                    tops.append(top)
                    texts.append(render(e.state, q, single).text)
                ps = probs(scorer, texts, [render(ex[0].state, {**ex[0].questions[head], "criteria": dict.fromkeys(t)},
                                                  single).codes for t in tops])
                res["stage2"] = metric(name, [rq.keys.index(t[int(np.argmax(p))]) for rq, t, p in zip(rqs, tops, ps)], y)

        table[name] = res
        out_path.write_text(json.dumps(table, indent=1))
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}, flush=True)

    keys = ["zero", "layout", "shot4", "shot8", "perm3", "stage2"]
    print("\nmean over tasks that have the method:")
    for k in keys:
        rows = [(t[k], t["zero"]) for t in table.values() if k in t]
        if rows:
            a, z = np.mean([r[0] for r in rows]), np.mean([r[1] for r in rows])
            print(f"  {k.ljust(7)} {a:.3f}  (zero on the same {len(rows)} tasks: {z:.3f})")


if __name__ == "__main__":
    main()
