"""Choice questions: score the option names themselves instead of single-letter codes.

log P(name | prompt) is the sum over the name's tokens. The prompt is run once and its cache is shared
by all options, so the extra cost is one short batched pass over the names.
Compares against the letter-code answers saved by the benchmark run on the same examples.
"""
import argparse
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.disable(logging.INFO)
from jev_benchmarking.cache import ResponseCache, request_key  # noqa: E402
from jev_benchmarking.tasks import get_tasks  # noqa: E402

HERE = Path(__file__).resolve().parent.parent


def render(state, q):
    lines = ["Answer the question about the following state.",
             "State:\n" + (state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)),
             "Question:\n" + str(q["instructions"]), "Options:"]
    for key, desc in q["criteria"].items():
        lines.append(f"- {key}" + (f" ({desc})" if desc else ""))
    lines.append("Reply with the name of the correct option only.")
    return "\n".join(lines)


class NameScorer:
    def __init__(self, name, vram):
        torch.cuda.set_per_process_memory_fraction(vram)
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id

    @torch.inference_mode()
    def score(self, text, names, chunk=64):
        out = self.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                           add_generation_prompt=True, enable_thinking=False)
        ids = torch.tensor([list(out["input_ids"] if hasattr(out, "keys") else out)], device="cuda")
        pre = self.model(input_ids=ids, use_cache=True)
        first = torch.log_softmax(pre.logits[0, -1].float(), -1)
        opts = [self.tok.encode(n, add_special_tokens=False) for n in names]
        total = torch.tensor([first[o[0]].item() for o in opts])
        multi = [i for i, o in enumerate(opts) if len(o) > 1]
        if multi and not hasattr(pre.past_key_values.layers[0], "batch_repeat_interleave"):
            return self.score_top(ids, opts, total).numpy()
        for s in range(0, len(multi), chunk):
            rows = multi[s:s + chunk]
            width = max(len(opts[i]) for i in rows)
            x = torch.full((len(rows), width), self.pad, device="cuda")
            m = torch.zeros((len(rows), width), dtype=torch.long, device="cuda")
            for r, i in enumerate(rows):
                x[r, :len(opts[i])] = torch.tensor(opts[i])
                m[r, :len(opts[i])] = 1
            cache = copy.deepcopy(pre.past_key_values)
            cache.batch_repeat_interleave(len(rows))
            mask = torch.cat([torch.ones((len(rows), ids.shape[1]), dtype=torch.long, device="cuda"), m], 1)
            lp = torch.log_softmax(self.model(input_ids=x, past_key_values=cache, attention_mask=mask).logits.float(), -1)
            for r, i in enumerate(rows):
                o = opts[i]
                total[i] += sum(lp[r, t - 1, o[t]].item() for t in range(1, len(o)))
        return total.numpy()

    def score_top(self, ids, opts, total, k=8):
        """For models whose cache cannot be copied across a batch: only the k best options by first token
        get their full name scored, each as its own prompt + name sequence. The rest keep a low score."""
        best = torch.topk(total, min(k, len(opts))).indices.tolist()
        n, width = ids.shape[1], max(len(opts[i]) for i in best)
        x = torch.full((len(best), n + width), self.pad, device="cuda")
        m = torch.zeros_like(x)
        for r, i in enumerate(best):
            x[r, :n] = ids[0]
            x[r, n:n + len(opts[i])] = torch.tensor(opts[i])
            m[r, :n + len(opts[i])] = 1
        lp = torch.log_softmax(self.model(input_ids=x, attention_mask=m).logits[:, n - 1:].float(), -1)
        out = torch.full_like(total, -1e9)
        for r, i in enumerate(best):
            out[i] = sum(lp[r, t, tok].item() for t, tok in enumerate(opts[i]))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tasks", nargs="+", default=["clinc150", "banking77", "financial_phrasebank", "ag_news",
                                                    "emotion", "anli"])
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--vram", type=float, default=0.8)
    args = ap.parse_args()

    scorer = NameScorer(args.model, args.vram)
    tag = f"hf:{args.model}"
    saved = ResponseCache(model=tag)
    res = {}
    for name in args.tasks:
        examples = get_tasks([name])[0].examples("eval", limit=args.limit)
        hit_name = hit_code = n = 0
        t0 = time.time()
        for e in examples:
            q = e.questions["answer"]
            keys = list(q["criteria"])
            lp = scorer.score(render(e.state, q), keys)
            hit_name += keys[int(np.argmax(lp))] == e.gold["answer"]
            old = saved.get(request_key(tag, e.state, e.questions))
            if old:
                hit_code += old["answers"]["answer"]["choice"] == e.gold["answer"]
            n += 1
        res[name] = {"n": n, "options": len(keys), "codes": hit_code / n, "names": hit_name / n,
                     "ms_per_decision": (time.time() - t0) / n * 1000}
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in res[name].items()}, flush=True)
    out = HERE / "results" / f"name_scoring_{args.model.split('/')[-1]}.json"
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
