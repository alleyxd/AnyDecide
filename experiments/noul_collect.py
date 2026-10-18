"""Collect yes/no scores for the calibration study.

For each binary task, every example is asked three ways: the question itself, a natural opposite
("is it negative?"), and a plain negation ("is it false that ... ?"). We store the Yes/No logit margin for each,
so every calibration method can be compared offline on exactly the same model outputs.

    python experiments/noul_collect.py Qwen/Qwen3-4B --n 1000 --vram 0.7
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anydecide import Decider, render

OUT = Path(__file__).resolve().parent.parent / "results" / "noul"


def pair(a, b):
    return f"Sentence 1: {a}\nSentence 2: {b}"


# name -> (loader, state(row), label(row) as 1 for "yes" to the main question, three question phrasings)
TASKS = {
    "sst2": (lambda: load_dataset("nyu-mll/glue", "sst2", split="validation"), lambda r: r["sentence"], lambda r: r["label"],
             ["Is the sentiment of this review positive?", "Is the sentiment of this review negative?",
              "Is it false that the sentiment of this review is positive?"]),
    "rotten_tomatoes": (lambda: load_dataset("cornell-movie-review-data/rotten_tomatoes", split="test"), lambda r: r["text"], lambda r: r["label"],
                        ["Is this movie review positive?", "Is this movie review negative?",
                         "Is it false that this movie review is positive?"]),
    "boolq": (lambda: load_dataset("google/boolq", split="validation"), lambda r: r["passage"], lambda r: int(r["answer"]),
              ["{q}?", "Based on the passage, is the answer to this question no: {q}?",
               "Is it false that the answer to this question is yes: {q}?"]),
    "sms_spam": (lambda: load_dataset("ucirvine/sms_spam", split="train"), lambda r: r["sms"], lambda r: r["label"],
                 ["Is this message spam?", "Is this a normal, legitimate message rather than spam?",
                  "Is it false that this message is spam?"]),
    "paws": (lambda: load_dataset("google-research-datasets/paws", "labeled_final", split="test"),
             lambda r: pair(r["sentence1"], r["sentence2"]), lambda r: r["label"],
             ["Do the two sentences mean the same thing?", "Do the two sentences mean different things?",
              "Is it false that the two sentences mean the same thing?"]),
    "mrpc": (lambda: load_dataset("nyu-mll/glue", "mrpc", split="test"), lambda r: pair(r["sentence1"], r["sentence2"]),
             lambda r: r["label"],
             ["Are the two sentences paraphrases of each other?", "Do the two sentences say different things?",
              "Is it false that the two sentences are paraphrases of each other?"]),
}


@torch.no_grad()
def margins(d: Decider, prompts: list[str], batch: int) -> list[float]:
    yes, no = d.code_id("Yes"), d.code_id("No")
    out = []
    for i in range(0, len(prompts), batch):
        enc = d.tok([d.prompt(p) for p in prompts[i:i + batch]], return_tensors="pt", padding=True,
                    add_special_tokens=False).to(d.model.device)
        logits = d.model(**enc, logits_to_keep=1).logits[:, -1].float()
        lp = torch.log_softmax(logits, -1)
        out += (lp[:, yes] - lp[:, no]).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--adapter")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--vram", type=float, default=0.7)
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    d = Decider(args.model, adapter=args.adapter, vram=args.vram, device=args.device)
    tag = (args.adapter or args.model).replace("/", "__").replace("\\", "__")
    out_dir = Path(args.out) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.tasks:
        path = out_dir / f"{name}.jsonl"
        if path.exists():
            print(f"skip {name}: done")
            continue
        load, state, label, phrasings = TASKS[name]
        rows = list(load())
        random.Random(0).shuffle(rows)
        rows = rows[: args.n]
        cols = []
        for ph in phrasings:
            prompts = []
            for r in rows:
                q = ph.format(q=r["question"]) if "{q}" in ph else ph
                prompts.append(render(state(r), {"type": "noul", "instructions": q}).text)
            cols.append(margins(d, prompts, args.batch))
        with path.open("w", encoding="utf-8") as f:
            for r, z0, z1, z2 in zip(rows, *cols):
                f.write(json.dumps({"y": int(label(r)), "z": z0, "z_opp": z1, "z_neg": z2}) + "\n")
        print(f"{name}: {len(rows)} examples -> {path}", flush=True)


if __name__ == "__main__":
    main()
