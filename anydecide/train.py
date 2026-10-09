"""Train any Hugging Face causal LM into a decision model with LoRA.

The training rows are prompt/completion pairs where the prompt is exactly what anydecide sends at inference and
the completion is the answer code. train/build_data.py makes such a file from public datasets; any file with the
same shape works, so you can train on your own decisions.

python -m anydecide.train --model Qwen/Qwen3-4B --data train/decisions_train.jsonl --out runs/decide-4b [--merge]
"""
import argparse
import json
import time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="JSONL with prompt (rendered question) and completion (answer code)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--merge", action="store_true", help="also save a merged model next to the adapter")
    args = ap.parse_args()

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    def fmt(r):
        prompt = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        return {"prompt": prompt, "completion": r["completion"] + (tok.eos_token or "")}

    ds = Dataset.from_list([fmt(r) for r in rows])
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
    lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
                    per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.accum,
                    lr_scheduler_type="cosine", warmup_steps=0.03, logging_steps=20, save_strategy="no", bf16=True,
                    gradient_checkpointing=True, report_to="none", max_length=args.max_len, completion_only_loss=True)
    t = time.time()
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
    result = trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    info = {"base": args.model, "examples": len(rows), "epochs": args.epochs, "lora_r": args.lora_r,
            "train_loss": round(result.training_loss, 4), "minutes": round((time.time() - t) / 60, 1)}
    if args.merge:
        merged = Path(args.out) / "merged"
        trainer.model.merge_and_unload().save_pretrained(merged)
        tok.save_pretrained(merged)
        info["merged"] = str(merged)
    Path(args.out, "train_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
