"""FastAPI service: /recommend (A/B-bucketed), /event (impression/click logging), /chat (guardrailed).
Run:  pip install fastapi uvicorn   &&   uvicorn fyp.api:app --reload
"""
import hashlib, json, logging, time
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .data import generate
from .pipeline import RecSys
from .safety import safe_chat

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("fyp")

app = FastAPI(title="FYP Lab")
ds = generate()
rs = RecSys(ds).fit()


def variant_for(user_id: int) -> str:
    """Deterministic 50/50 bucketing: same user always sees the same variant."""
    return "treatment" if int(hashlib.md5(str(user_id).encode()).hexdigest(), 16) % 2 else "control"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = Query(10, ge=1, le=50)):
    if not 0 <= user_id < ds.n_users:
        raise HTTPException(404, "unknown user")
    t0 = time.perf_counter()
    variant = variant_for(user_id)
    if user_id in set(ds.cold_users.tolist()):
        items, path = rs.recommend_cold(ds.fav_cats[user_id], k), "cold_start"
    elif variant == "treatment":
        items, path = rs.recommend_ranked([user_id], k, mmr_lambda=0.7)[user_id], "ranker_mmr"
    else:
        items, path = rs.recommend_scores([user_id], "mf", k)[user_id], "mf"
    ms = (time.perf_counter() - t0) * 1000
    log.info(json.dumps({"event": "impression", "user": user_id, "variant": variant,
                         "path": path, "items": items, "latency_ms": round(ms, 2)}))
    return {"user_id": user_id, "variant": variant, "path": path, "items": items, "latency_ms": round(ms, 2)}


class Event(BaseModel):
    user_id: int
    item_id: int
    type: str            # impression | click | skip | like | watch
    value: float = 1.0


@app.post("/event")
def event(e: Event):
    log.info(json.dumps({"event": e.type, **e.model_dump(), "variant": variant_for(e.user_id), "ts": time.time()}))
    return {"ok": True}


class ChatIn(BaseModel):
    message: str


def echo_llm(prompt: str) -> str:       # replace with a real LLM call (e.g. Groq)
    return f"(demo yanıtı) Mesajını aldım: {prompt}"


@app.post("/chat")
def chat(body: ChatIn):
    return safe_chat(body.message, echo_llm)
