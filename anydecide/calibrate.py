"""Make a question's probabilities trustworthy with a few labelled examples.

Untrained chat models put near-certain probability on almost every answer, and lean towards some options.
For a yes/no question we fit Platt scaling on the Yes/No margin; for choice and score questions, one temperature
plus a bias per option. Both are fitted on the model's own code probabilities, so the cost is one forward pass
per labelled example, and a fit is a handful of numbers that can be saved next to the question.

    cal = calibrate.fit(decider, question, [(state, "billing"), (state2, "technical"), ...])
    cal.save("department.json")
    decider.decide(state, {"department": question}, calibration={"department": cal})
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


@dataclass
class Calibration:
    kind: str                                         # "noul", "choice" or "score"
    a: float = 1.0                                    # noul: p = sigmoid(a * (lp_yes - lp_no) + b)
    b: float = 0.0
    temperature: float = 1.0                          # choice / score: softmax(lp / temperature + bias)
    bias: list[float] = field(default_factory=list)
    n: int = 0

    def apply(self, logprobs: torch.Tensor) -> list[float]:
        if self.kind == "noul":
            p = torch.sigmoid(self.a * (logprobs[0] - logprobs[1]) + self.b).item()
            return [p, 1 - p]
        bias = torch.tensor(self.bias or [0.0] * len(logprobs), dtype=logprobs.dtype)
        return torch.softmax(logprobs / self.temperature + bias, -1).tolist()

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def fit_yes_no(margins: list[float], labels: list[int], l2: float = 1e-2) -> tuple[float, float]:
    """Platt scaling: logistic regression of the label on the Yes/No margin (Newton's method)."""
    x = torch.tensor(margins, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    X = torch.stack([x, torch.ones_like(x)], 1)
    w = torch.zeros(2, dtype=torch.float64)   # start at p = 0.5: from a = 1 the curvature can vanish
    reg = torch.tensor([l2, 0.0], dtype=torch.float64)
    for _ in range(50):
        p = torch.sigmoid(X @ w)
        grad = X.T @ (p - y) + reg * w
        hess = (X * (p * (1 - p))[:, None]).T @ X + torch.diag(reg + 1e-9)
        step = torch.linalg.solve(hess, grad).clamp(-5, 5)
        w = w - step
        if step.abs().max() < 1e-8:
            break
    return w[0].item(), w[1].item()


def fit_choice(logprob_rows: list[list[float]], gold: list[int], l2: float = 1e-2) -> tuple[float, list[float]]:
    """A temperature and one bias per option that minimise log loss (the biases are kept centred)."""
    lp = torch.tensor(logprob_rows, dtype=torch.float64)
    y = torch.tensor(gold)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(lp.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t, bias], max_iter=200, line_search_fn="strong_wolfe")

    def loss():
        opt.zero_grad()
        z = lp / log_t.exp() + (bias - bias.mean())
        out = torch.nn.functional.cross_entropy(z, y) + l2 * (bias ** 2).mean()
        out.backward()
        return out

    opt.step(loss)
    b = (bias - bias.mean()).detach()
    return log_t.exp().item(), b.tolist()


def fit(decider, question: dict, examples: list[tuple], batch: int = 16) -> Calibration:
    """examples: (state, right answer) pairs, where the answer is an option key, a level index, or True/False."""
    kind = question["type"]
    rows, gold = [], []
    for i in range(0, len(examples), batch):
        chunk = examples[i:i + batch]
        rows += decider.code_logprobs([s for s, _ in chunk], question)
        for _, ans in chunk:
            if kind == "noul":
                gold.append(int(bool(ans) and str(ans).lower() not in ("no", "false", "0")))
            elif kind == "choice":
                gold.append(list(question["criteria"]).index(ans))
            else:
                gold.append(int(ans))
    if kind == "noul":
        a, b = fit_yes_no([r[0] - r[1] for r in rows], gold)
        return Calibration("noul", a=a, b=b, n=len(rows))
    t, bias = fit_choice(rows, gold)
    return Calibration(kind, temperature=t, bias=bias, n=len(rows))
