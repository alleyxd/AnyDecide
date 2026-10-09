"""Latency of the decision library on one GPU: one question, several questions about the same text (batched into
one forward pass), and the same answer generated as JSON text for comparison.

python bench_speed.py MODEL [--adapter PATH]
"""
import argparse
import statistics
import time

import torch

from anydecide import Decider

STATE = ("Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan. "
         "Invoice INV-20391, account 88213, plan Team, 14 seats.")
QS = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages",
                                "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    "refund": {"type": "noul", "instructions": "Is the user asking for money back?"},
    "language": {"type": "choice", "instructions": "Which language is the message in?",
                 "criteria": {"English": None, "French": None, "Spanish": None}},
    "sentiment": {"type": "choice", "instructions": "What is the tone?",
                  "criteria": {"angry": None, "neutral": None, "happy": None}},
}


def timed(fn, n=30):
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t) * 1000)
    return statistics.median(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--adapter")
    args = ap.parse_args()
    d = Decider(args.model, args.adapter, vram=0.7)
    print(d.decide(STATE, QS))
    one = timed(lambda: d.decide(STATE, {"department": QS["department"]}))
    six = timed(lambda: d.decide(STATE, QS))
    p = d.prompt("Classify this ticket and reply as JSON with keys department, urgency, churn_risk, refund, language, "
                 "sentiment.\n\n" + STATE)
    enc = d.tok(p, return_tensors="pt", add_special_tokens=False).to(d.model.device)
    gen = timed(lambda: d.model.generate(**enc, max_new_tokens=80, do_sample=False), n=5)
    print(f"{args.model}: 1 question {one:.1f} ms | 6 questions in one batch {six:.1f} ms | "
          f"same 6 answers generated as JSON {gen:.0f} ms")
