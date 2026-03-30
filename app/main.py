import os
import numpy as np
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
import google.generativeai as genai

from app.rrf_core import *
from app.embeddings import get_embedding
from app.rate_limit import rate_limiter
from app.logger import log_request

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

app = FastAPI(title="RRF API PRO", version="2.0")

class Input(BaseModel):
    prompt: str

@app.get("/")
def root():
    return {"status": "RRF PRO running 🚀"}

@app.post("/generate", dependencies=[Depends(rate_limiter)])
def generate(data: Input):
    response = model.generate_content(data.prompt)
    base_text = response.text

    chunks = [c.strip() for c in base_text.split('.') if c.strip()]
    embeddings = np.array([get_embedding(c) for c in chunks])

    base_score = rrf_score(base_text, embeddings)

    new_embeddings = rrf_transform(embeddings)
    norms = [np.linalg.norm(e) for e in new_embeddings]

    sorted_chunks = [x for _, x in sorted(zip(norms, chunks), reverse=True)]
    rrf_text = ". ".join(sorted_chunks)

    new_score = rrf_score(rrf_text, new_embeddings)

    result = {
        "baseline": base_score,
        "rrf": new_score,
        "improvement": new_score - base_score,
        "output": rrf_text
    }

    log_request(result)

    return result

@app.post("/analyze", dependencies=[Depends(rate_limiter)])
def analyze(data: Input):
    text = data.prompt

    chunks = [c.strip() for c in text.split('.') if c.strip()]
    embeddings = np.array([get_embedding(c) for c in chunks])

    return {
        "score": rrf_score(text, embeddings),
        "entropy": entropy(text)
    }
