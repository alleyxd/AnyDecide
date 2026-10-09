"""Download a model, run it on the benchmark tasks, keep the scores, delete the weights.

Usage: python run_models.py [--limit 300] [--keep] model [model ...]
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import scan_cache_dir, snapshot_download

HERE = Path(__file__).resolve().parent
BENCH = HERE / "ref" / "jev-benchmarking"
OUT = HERE / "results" / "baselines"

TASKS = ["sst2", "ag_news", "banking77", "boolq", "clinc150", "toxic_chat", "emotion", "paws", "anli",
         "financial_phrasebank", "rotten_tomatoes", "sms_spam"]

# set the VRAM cap, then hand over to the benchmark's own runner
LAUNCH = ("import sys, runpy, torch; torch.cuda.set_per_process_memory_fraction({vram}); "
          "sys.argv = ['run_open_model.py'] + sys.argv[1:]; runpy.run_path('scripts/run_open_model.py', run_name='__main__')")


def run(cmd, log):
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        return subprocess.run(cmd, cwd=BENCH, stdout=f, stderr=subprocess.STDOUT).returncode


def delete_weights(model):
    for repo in scan_cache_dir().repos:
        if repo.repo_id == model and repo.repo_type == "model":
            shutil.rmtree(repo.repo_path)
            return repo.size_on_disk
    return 0


def bench(model, limit, vram, keep):
    local = (BENCH / model).exists()   # a model folder saved under ref/jev-benchmarking, e.g. models/x
    tag = model.replace("/", "__")
    out = OUT / tag
    if (out / "summary.json").exists():
        print("skip", model)
        return
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"

    if not local:
        snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    runs = BENCH / "runs" / tag
    lim = ["--limit", str(limit)] if limit else []
    t0 = time.time()
    code = run([sys.executable, "-c", LAUNCH.format(vram=vram), *TASKS, "--model", model, "--out", str(runs),
                "--max-batch-tokens", "4096", *lim], log)
    seconds = time.time() - t0
    if code == 0:
        code = run([sys.executable, "scripts/import_responses.py", *map(str, runs.glob("*.jsonl"))], log)
    if code == 0:
        code = run([sys.executable, "scripts/evaluate.py", *TASKS, "--no-ci", "--model", f"hf:{model}", *lim], log)

    summary = {"model": model, "limit": limit, "seconds": round(seconds), "ok": code == 0, "tasks": {}}
    if code == 0:
        n = 0
        for f in (BENCH / "results" / "eval" / "open_models" / tag).glob("*.json"):
            r = json.loads(f.read_text())
            shutil.copy(f, out / f.name)
            n += r["n_answered"]
            ece = next((h["ece"] for h in r["heads"].values() if "ece" in h), None)
            summary["tasks"][r["task"]] = {"metric": r["primary"]["metric"], "value": r["primary"]["value"],
                                           "ece": ece, "n": r["n_answered"]}
        summary["examples"] = n
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if not keep and not local:
        freed = delete_weights(model)
        print(f"deleted {model} ({freed / 1e9:.1f} GB)")
    print(model, "ok" if code == 0 else "FAILED, see " + str(log), f"{seconds:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--vram", type=float, default=0.8)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    for m in args.models:
        bench(m, args.limit, args.vram, args.keep)
