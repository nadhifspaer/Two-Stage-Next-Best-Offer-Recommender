# FastAPI /recommend, /health and /metrics, serving the committed offer store through src/score_pipeline.py
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

import src.score_pipeline as score_pipeline

REQUEST_COUNT = Counter("api_requests_total", "Total requests received", ["endpoint"])
ERROR_COUNT = Counter("api_errors_total", "Total error responses", ["endpoint"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds", ["endpoint"])

# series start at zero so the first error after startup is visible to rate()
for _endpoint in ("/health", "/recommend/{customer_id}", "/metrics"):
    REQUEST_COUNT.labels(endpoint=_endpoint)
    ERROR_COUNT.labels(endpoint=_endpoint)


@asynccontextmanager
async def lifespan(app: FastAPI):
    score_pipeline.get_default_store()
    yield


app = FastAPI(title="H&M Next Best Offer API", lifespan=lifespan)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        route = request.scope.get("route")
        endpoint = route.path if route is not None else request.url.path
        REQUEST_COUNT.labels(endpoint=endpoint).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
        if response is None or response.status_code >= 400:
            ERROR_COUNT.labels(endpoint=endpoint).inc()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/{customer_id}")
def recommend(customer_id: str, k: int = Query(12, ge=1, le=12)):
    result = score_pipeline.recommend(customer_id, k=k)
    if result is None:
        raise HTTPException(status_code=404, detail="customer_id not found in offer store")
    return result


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
