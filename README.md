# AnyDecide

[![tests](https://github.com/alleyxd/AnyDecide/actions/workflows/tests.yml/badge.svg)](https://github.com/alleyxd/AnyDecide/actions/workflows/tests.yml)

Turn a local language model into a fast decision API. Instead of generating text you have to parse, you ask typed
questions (pick one option, give a score, yes or no) and get typed answers with probabilities back. All questions
about one input run in a single batched forward pass, and each answer is read straight from the next-token
probabilities, so the output always has the requested type and there is nothing to parse.

```python
from anydecide import Decider

d = Decider("Qwen/Qwen3-4B")
d.decide("We were billed twice for March. Refund it today or we cancel.", {
    "department": {"type": "choice", "instructions": "Which team handles this?",
                   "criteria": {"billing": "payments, refunds", "technical": "bugs", "other": None}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "soon", "blocking"]},
    "churn": {"type": "noul", "instructions": "Does the customer threaten to leave?"},
})
```

```python
{"department": {"type": "choice", "choice": "billing",
                "probabilities": {"billing": 0.98, "technical": 0.01, "other": 0.01}, "confidence": 0.97},
 "urgency": {"type": "score", "score": 1.81, "probabilities": {"0": 0.01, "1": 0.18, "2": 0.81}, ...},
 "churn": {"type": "noul", "noul": 0.88}}
```

As an HTTP service:

```bash
ANYDECIDE_MODEL=Qwen/Qwen3-4B uvicorn anydecide.server:app --port 8009
curl localhost:8009/v1/decide -H 'content-type: application/json' -d '{"state": "...", "questions": {...}}'
```

## Speed

`python bench_speed.py MODEL`, RTX 5090, a short support ticket and six questions about it:

| Model | AnyDecide, 6 questions in one pass | Same model writing the 6 answers as JSON | Speed-up |
|---|---|---|---|
| Qwen3-4B | 57 ms | 2,543 ms | 45x |
| decision-tuned Qwen3-4B | 57 ms | 1,138 ms | 20x |

A single question takes 46 to 50 ms, so six answers cost about as much as one: they share a single forward pass.
The tuned model is faster at writing JSON only because it writes shorter answers; with AnyDecide both models take
the same time.

## Accuracy

Twelve classification and routing tasks (sentiment, intent routing with up to 150 intents, spam, toxicity, yes/no
reading questions, entailment, paraphrase, news topics, emotion), 300 examples each, same prompts and same samples
for every model. Score = accuracy, or F1 for spam and toxic-chat detection.

| Model | Mean of 12 tasks |
|---|---|
| **Qwen3-4B + decision LoRA (this repo)** | **0.756** |
| Qwen3.5-9B | 0.781 |
| Qwen3.5-4B | 0.747 |
| Qwen2.5-7B-Instruct | 0.746 |
| Qwen3-4B | 0.713 |
| Mistral-7B-Instruct-v0.3 | 0.667 |
| Llama-3.1-8B-Instruct | 0.652 |
| Phi-4-mini | 0.611 |
| Qwen3.5-2B | 0.571 |
| Qwen3-1.7B | 0.535 |
| Qwen3-0.6B | 0.433 |

Every model except the first is used as-is, with no training. Any of them works with `Decider` directly; training
makes a model better at it (next section).

## Calibrating with a few labels

Untrained chat models are far too sure of themselves, and many lean towards one answer: a small model asked whether
a review is positive says yes with 0.98 probability on average. A few dozen labelled examples fix most of that
without training. Yes/no questions get Platt scaling on the Yes/No margin; choice and score questions get a
temperature and a bias per option. A fit is a handful of numbers you can save next to the question.

```python
from anydecide import Calibration, calibrate

q = {"type": "noul", "instructions": "Is the sentiment of this review positive?"}
cal = calibrate.fit(d, q, [(review, label) for review, label in labelled[:60]])
cal.save("positive.json")

d.decide(review, {"positive": q}, calibration={"positive": Calibration.load("positive.json")})
```

On SST-2 with Qwen3-0.6B, fitting on 60 reviews and testing on 100 others: calibration error 0.428 to 0.159,
accuracy 0.55 to 0.60. For the HTTP service, point `ANYDECIDE_CALIBRATION` at a JSON file of
`{question name: saved calibration}`.

## Training a decision model

`train/build_data.py` builds 36,194 decision questions from 15 public datasets that do **not** appear in the
benchmark (DBpedia, Yahoo Answers, Amazon and Yelp reviews, SNLI, MNLI, QQP, QNLI, four TweetEval tasks, MASSIVE
intents, Civil Comments, financial tweets, 20 Newsgroups). Each one is rendered with exactly the prompt used at
inference, and the target is the single answer code. Training is built in:

```bash
uv pip install -e ".[train]"
python train/build_data.py                                   # or your own prompt/completion JSONL
python -m anydecide.train --model Qwen/Qwen3-4B --data train/decisions_train.jsonl --out runs/decide-4b --merge
```

```python
d = Decider("runs/decide-4b/merged")                         # or Decider("Qwen/Qwen3-4B", adapter="runs/decide-4b")
```

Any Hugging Face chat model can be trained this way, as long as the answer codes (A, B, ..., Yes, No, 0 to 9) are
single tokens in its tokenizer, which holds for the common families; `Decider` raises a clear error otherwise. The
numbers below are for Qwen3-4B, the only model I have trained so far.

To train on your own decisions, write rows as `{"prompt": anydecide.render(state, question).text, "completion":
"<code>"}`, where the code is the letter of the right option, `Yes`/`No`, or the level number.

One epoch, LoRA rank 16, batch 4 with 4 accumulation steps: 74 minutes on an RTX 5090, 10.6 GB peak memory. Per task,
against the same model untrained:

| Task | Qwen3-4B | + decision LoRA |
|---|---|---|
| 150-intent routing (CLINC) | 0.593 | **0.793** |
| SMS spam (F1) | 0.553 | **0.776** |
| Financial sentiment | 0.737 | **0.837** |
| Banking intents | 0.610 | **0.683** |
| Yes/no reading questions (BoolQ) | 0.853 | **0.880** |
| SST-2 sentiment | 0.947 | **0.963** |
| Movie reviews | 0.883 | **0.893** |
| Emotion | 0.540 | **0.553** |
| News topics | 0.877 | 0.877 |
| Toxic chat (F1) | 0.750 | 0.737 |
| NLI (ANLI) | 0.543 | 0.503 |
| Paraphrase (PAWS) | 0.667 | 0.573 |
| **Mean** | **0.713** | **0.756** |
| Mean calibration error (ECE) | 0.241 | **0.108** |

Training helped most where the untrained model was badly biased (routing among 150 intents, spam) and more than
halved the calibration error: the untrained model says 100% about almost everything, the tuned one gives
probabilities you can set thresholds on.

It got worse on paraphrase detection. PAWS is built from sentence pairs with the same words in a different order,
and the training set's paraphrase data (Quora question pairs) rewards word overlap, which is exactly the shortcut
PAWS punishes. The fix is better training data, not more of it.

## What did not work

Before training, I tried to improve the untrained models without it. These are measured on the same benchmark
(Qwen3.5-4B unless stated), and they are here so nobody has to repeat them:

| Idea | Effect on the 12-task mean |
|---|---|
| Label-free prior correction (divide by each option's average probability) | worse: hurts imbalanced tasks such as spam |
| Bias and threshold fitted on 150 labelled examples per task | +4 points (0.747 to 0.787), needs labels |
| Scoring option names instead of letter codes | worse on 4 of 5 tasks, and 3 to 4 times slower |
| 4 or 8 worked examples in the prompt | +1.5 / +2.5 points, very uneven across tasks |
| Averaging three option orders | +1.6 points on the 8 choice tasks |
| Two-stage choice for 100+ options | +5 points on the 150-intent task, none on Banking77 |

Training gave by far the largest gain, though not on every task (see the table above).

## Paper

[Decisions in One Forward Pass: Typed Answers from Local Language Models, and Making Their Yes/No Probabilities Mean
Something](paper/anydecide.pdf) covers the speed and accuracy results above and a calibration study of yes/no answers
across five models and six tasks. The short version: averaging a question with its opposite makes the answers coherent
but not calibrated, while Platt scaling on 25 labelled examples more than halves calibration error (0.197 to 0.089)
and raises accuracy from 0.779 to 0.845. `experiments/run_noul.sh` reproduces the study; every model score it used is in
`results/noul/`.

```bibtex
@misc{ashter2026anydecide,
  title  = {Decisions in One Forward Pass: Typed Answers from Local Language Models, and Making Their Yes/No
            Probabilities Mean Something},
  author = {Ashter, Ali},
  year   = {2026},
  url    = {https://github.com/alleyxd/AnyDecide}
}
```

## Tests

```bash
pytest -q
```

One test checks that the prompts are byte-for-byte identical to the benchmark's, so scores stay comparable.

## Reproducing the benchmark

The tasks and the scoring harness come from the open
[jev-benchmarking](https://github.com/AppliedMachineLearning-Lab/jev-benchmarking) project (MIT), which is not
included here. To rerun the tables:

```bash
git clone https://github.com/AppliedMachineLearning-Lab/jev-benchmarking ref/jev-benchmarking
uv pip install -e ref/jev-benchmarking
python run_models.py Qwen/Qwen3-4B --keep          # downloads, scores 12 tasks x 300, writes results/baselines/
```

`results/baselines/` holds every per-task score behind the tables above.

## License

MIT
