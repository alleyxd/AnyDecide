"""Run the benchmark tasks against a local /v1/decide server and save answers in the benchmark's format."""
import argparse
import json
import logging
import time
import urllib.request
from pathlib import Path

logging.disable(logging.INFO)
from jev_benchmarking.cache import request_key  # noqa: E402
from jev_benchmarking.tasks import get_tasks  # noqa: E402

TASKS = ["sst2", "ag_news", "banking77", "boolq", "clinc150", "toxic_chat", "emotion", "paws", "anli",
         "financial_phrasebank", "rotten_tomatoes", "sms_spam"]


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="model tag for the response cache, e.g. hf:Qwen/Qwen3-4B")
    ap.add_argument("--url", default="http://127.0.0.1:8009/v1/decide")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {json.loads(l)["key"] for l in args.out.read_text(encoding="utf-8").splitlines() if l.strip()}
    with args.out.open("a", encoding="utf-8") as f:
        for name in args.tasks:
            t0, n = time.time(), 0
            for e in get_tasks([name])[0].examples("eval", limit=args.limit):
                key = request_key(args.tag, e.state, e.questions)
                if key in done:
                    continue
                t = time.time()
                r = post(args.url, {"state": e.state, "questions": e.questions})
                r["model"] = args.tag
                f.write(json.dumps({"key": key, "task": name, "uid": e.uid, "latency_s": time.time() - t,
                                    "response": r}, ensure_ascii=False) + "\n")
                done.add(key)
                n += 1
            f.flush()
            if n:
                print(f"{name}: {n} requests, {(time.time() - t0) / n * 1000:.0f} ms each", flush=True)


if __name__ == "__main__":
    main()
