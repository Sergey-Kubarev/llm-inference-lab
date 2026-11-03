
import time

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="LLM Inference Lab")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:0.6b"


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(request: GenerateRequest):
    payload = {
        "model": MODEL,
        "prompt": request.prompt,
        "stream": False,
        "think": False,
        "options": {
            "num_predict": 128,
            "temperature": 0
        }
    }

    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()

    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama request failed: {type(exc).__name__}"
        ) from exc

    elapsed = time.perf_counter() - start
    data = response.json()

    tokens = data.get("eval_count", 0)
    eval_seconds = data.get("eval_duration", 0) / 1e9

    return {
        "model": MODEL,
        "answer": data.get("response", ""),
        "metrics": {
            "latency_seconds": round(elapsed, 3),
            "output_tokens": tokens,
            "tokens_per_second": round(
                tokens / eval_seconds, 2
            ) if eval_seconds > 0 else None,
            "model_load_seconds": round(
                data.get("load_duration", 0) / 1e9, 3
            )
        }
    }
