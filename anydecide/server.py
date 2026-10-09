"""HTTP endpoint: POST /v1/decide {"state": ..., "questions": {...}}.

ANYDECIDE_MODEL=Qwen/Qwen3-4B [ANYDECIDE_ADAPTER=path] uvicorn anydecide.server:app --port 8009
"""
import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model import Decider

app = FastAPI(title="anydecide")
decider = Decider(os.environ.get("ANYDECIDE_MODEL", "Qwen/Qwen3-4B"), os.environ.get("ANYDECIDE_ADAPTER"),
                  temperature=float(os.environ.get("ANYDECIDE_TEMPERATURE", "1.0")))


class Request(BaseModel):
    state: str | dict | list
    questions: dict


@app.post("/v1/decide")
def decide(req: Request):
    t = time.perf_counter()
    try:
        out = decider.decide(req.state, req.questions)
    except (KeyError, ValueError) as e:
        raise HTTPException(422, str(e))
    out["latency_ms"] = round((time.perf_counter() - t) * 1000, 1)
    return out
