"""Decision-format training data from public datasets that are NOT in the benchmark.

Every example is rendered with the benchmark's own prompt (state, question, lettered options or Yes/No) and the
target is the single answer code, so a model trained on this is scored exactly the way it was trained.
Writes train/decisions_train.jsonl in tunekit's prompt/completion format.
"""
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ref" / "jev-benchmarking"))
from jev_benchmarking.openmodel.prompts import render  # noqa: E402

rng = random.Random(0)
PER_SOURCE = 2500


def label_names(ds, col="label"):
    return ds.features[col].names


def balanced(ds, col, n):
    """Up to n rows with labels spread as evenly as the data allows."""
    by = {}
    for i, y in enumerate(ds[col]):
        by.setdefault(y, []).append(i)
    for v in by.values():
        rng.shuffle(v)
    picked, k = [], 0
    while len(picked) < n and any(by.values()):
        for y in list(by):
            if by[y]:
                picked.append(by[y].pop())
        k += 1
    return ds.select(picked[:n])


def choice(state, instructions, options, gold):
    return {"state": state, "q": {"type": "choice", "instructions": instructions, "criteria": options}, "gold": gold}


def noul(state, instructions, gold: bool):
    return {"state": state, "q": {"type": "noul", "instructions": instructions}, "gold": gold}


def sources():
    ds = balanced(load_dataset("fancyzhx/dbpedia_14", split="train"), "label", PER_SOURCE)
    names = label_names(ds)
    yield "dbpedia", [choice(r["title"] + ": " + r["content"], "What kind of entity does this encyclopedia entry describe?",
                             {n: None for n in names}, names[r["label"]]) for r in ds]

    ds = balanced(load_dataset("community-datasets/yahoo_answers_topics", split="train"), "topic", PER_SOURCE)
    names = label_names(ds, "topic")
    yield "yahoo", [choice({"question": r["question_title"], "answer": r["best_answer"][:600]},
                           "Which topic does this question belong to?", {n: None for n in names}, names[r["topic"]])
                    for r in ds]

    ds = balanced(load_dataset("fancyzhx/amazon_polarity", split="train"), "label", PER_SOURCE)
    yield "amazon", [noul(r["title"] + ". " + r["content"], "Is this product review positive?", r["label"] == 1) for r in ds]

    ds = balanced(load_dataset("fancyzhx/yelp_polarity", split="train"), "label", PER_SOURCE)
    yield "yelp", [choice(r["text"][:1500], "What is the sentiment of this restaurant review?",
                          {"negative": None, "positive": None}, ["negative", "positive"][r["label"]]) for r in ds]

    nli = {"entailment": "the hypothesis follows from the premise", "neutral": "it may or may not be true",
           "contradiction": "the hypothesis contradicts the premise"}
    for name, split, cfg in (("snli", "train", None), ("mnli", "train", "mnli")):
        ds = load_dataset("stanfordnlp/snli" if name == "snli" else "nyu-mll/glue", cfg, split=split)
        ds = balanced(ds.filter(lambda r: r["label"] in (0, 1, 2)), "label", PER_SOURCE)
        keys = list(nli)
        yield name, [choice({"premise": r["premise"], "hypothesis": r["hypothesis"]},
                            "How does the hypothesis relate to the premise?", nli, keys[r["label"]]) for r in ds]

    ds = balanced(load_dataset("nyu-mll/glue", "qqp", split="train"), "label", PER_SOURCE)
    yield "qqp", [noul({"question_1": r["question1"], "question_2": r["question2"]},
                       "Do the two questions ask the same thing?", r["label"] == 1) for r in ds]

    ds = balanced(load_dataset("nyu-mll/glue", "qnli", split="train"), "label", PER_SOURCE)
    yield "qnli", [noul({"question": r["question"], "sentence": r["sentence"]},
                        "Does the sentence contain the answer to the question?", r["label"] == 0) for r in ds]

    for cfg, instr in (("offensive", "Is this tweet offensive?"), ("hate", "Does this tweet express hate speech?"),
                       ("irony", "Is this tweet ironic?")):
        ds = balanced(load_dataset("cardiffnlp/tweet_eval", cfg, split="train"), "label", PER_SOURCE // 2)
        yield f"tweet_{cfg}", [noul(r["text"], instr, r["label"] == 1) for r in ds]
    ds = balanced(load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train"), "label", PER_SOURCE)
    names = label_names(ds)
    yield "tweet_sentiment", [choice(r["text"], "What is the sentiment of this tweet?", {n: None for n in names},
                                     names[r["label"]]) for r in ds]

    ds = load_dataset("mteb/amazon_massive_intent", "en", split="train").shuffle(seed=0).select(range(PER_SOURCE))
    intents = sorted(set(load_dataset("mteb/amazon_massive_intent", "en", split="train")["label"]))
    yield "massive", [choice(r["text"], "Which intent does this request to a voice assistant express?",
                             {i: None for i in intents}, r["label"]) for r in ds]

    cc = load_dataset("google/civil_comments", split="train").select(range(200_000))
    tox = cc.filter(lambda r: r["toxicity"] >= 0.5).select(range(PER_SOURCE // 2))
    ok = cc.filter(lambda r: r["toxicity"] < 0.1).shuffle(seed=0).select(range(PER_SOURCE // 2))
    yield "civil_comments", [noul(r["text"][:1500], "Is this comment toxic, insulting or hateful?", r["toxicity"] >= 0.5)
                             for r in list(tox) + list(ok)]

    ds = balanced(load_dataset("zeroshot/twitter-financial-news-sentiment", split="train"), "label", PER_SOURCE)
    keys = ["bearish", "bullish", "neutral"]
    yield "fin_tweets", [choice(r["text"], "Is this financial news bearish, bullish or neutral for the asset?",
                                {k: None for k in keys}, keys[r["label"]]) for r in ds]

    ds = balanced(load_dataset("SetFit/20_newsgroups", split="train"), "label", PER_SOURCE)
    groups = sorted(set(ds["label_text"]))
    yield "newsgroups", [choice(r["text"][:1500], "Which newsgroup was this post written for?", {g: None for g in groups},
                                r["label_text"]) for r in ds if r["text"].strip()]


def main():
    out = []
    for name, items in sources():
        rows = []
        for it in items:
            rq = render(it["state"], it["q"])
            code = ("Yes" if it["gold"] else "No") if rq.kind == "noul" else rq.codes[rq.keys.index(str(it["gold"]))]
            rows.append({"prompt": rq.text, "completion": code, "source": name})
        print(f"{name}: {len(rows)}", flush=True)
        out += rows
    rng.shuffle(out)
    path = HERE / "decisions_train.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(out), "examples ->", path)


if __name__ == "__main__":
    main()
