#!/usr/bin/env bash
# Reproduces the yes/no calibration study in the paper: collect scores for each model, then analyse.
# The tuned model needs runs/decide-4b from: python -m anydecide.train --model Qwen/Qwen3-4B --data train/decisions_train.jsonl --out runs/decide-4b
set -e
cd "$(dirname "$0")/.."
for m in Qwen/Qwen3-4B Qwen/Qwen3-8B meta-llama/Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3; do
  python experiments/noul_collect.py "$m" --n 1000
done
[ -d runs/decide-4b ] && python experiments/noul_collect.py Qwen/Qwen3-4B --adapter runs/decide-4b --n 1000
python experiments/noul_analyze.py
