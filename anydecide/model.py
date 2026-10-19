import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .questions import answer, render


class Decider:
    """Any Hugging Face causal LM as a decision model. All questions about one state go through the model in a single
    batched forward pass, and probabilities come from the next-token logits restricted to the answer codes: no
    text is generated, and the output always has the requested type."""

    def __init__(self, model: str, adapter: str | None = None, device: str = "cuda", temperature: float = 1.0,
                 vram: float | None = None):
        if vram and device.startswith("cuda"):
            torch.cuda.set_per_process_memory_fraction(vram)
        self.tok = AutoTokenizer.from_pretrained(adapter or model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map=device)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload()
        self.model.eval()
        self.temperature = temperature
        self._code_ids = {}

    def code_id(self, code: str) -> int:
        if code not in self._code_ids:
            ids = self.tok.encode(code, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"answer code {code!r} is not a single token for this tokenizer")
            self._code_ids[code] = ids[0]
        return self._code_ids[code]

    def prompt(self, text: str) -> str:
        return self.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)

    @torch.no_grad()
    def _logprobs(self, rendered: list) -> list[torch.Tensor]:
        """One forward pass over the rendered questions; the log-probabilities of each one's answer codes."""
        enc = self.tok([self.prompt(r.text) for r in rendered], return_tensors="pt", padding=True,
                       add_special_tokens=False).to(self.model.device)
        logits = self.model(**enc, logits_to_keep=1).logits[:, -1].float()
        lp = torch.log_softmax(logits, -1)
        return [lp[row, [self.code_id(c) for c in r.codes]].cpu() for row, r in enumerate(rendered)]

    def code_logprobs(self, states: list, question: dict) -> list[list[float]]:
        """The same question about several states, batched; used to fit a calibration."""
        return [lp.tolist() for lp in self._logprobs([render(s, question) for s in states])]

    def decide(self, state, questions: dict, calibration: dict | None = None) -> dict:
        """calibration: optional {question name: anydecide.calibrate.Calibration} for some or all questions."""
        keys = list(questions)
        rendered = [render(state, questions[k]) for k in keys]
        out = {}
        for k, r, lp in zip(keys, rendered, self._logprobs(rendered)):
            cal = (calibration or {}).get(k)
            probs = cal.apply(lp) if cal else torch.softmax(lp / self.temperature, -1).tolist()
            out[k] = answer(r, probs, questions[k])
        return {"answers": out}


def fit_temperature(logprob_rows: list[list[float]], gold: list[int]) -> float:
    """Single temperature that minimises log loss on labelled examples (grid search, robust and dependency-free)."""
    def nll(t):
        total = 0.0
        for lp, g in zip(logprob_rows, gold):
            z = [x / t for x in lp]
            m = max(z)
            total -= z[g] - (m + math.log(sum(math.exp(x - m) for x in z)))
        return total / len(gold)
    grid = [0.5 + 0.05 * i for i in range(91)]   # 0.5 .. 5.0
    return min(grid, key=nll)
