from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __author__
from .model_service import default_model


app = FastAPI(
    title="Projet 11 - API Sentiment Politique",
    description="API de classification de sentiment pour des tweets politiques.",
    version="1.0.0",
)

service = default_model()


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, example="The election results show a clear victory for democracy.")


class PredictResponse(BaseModel):
    label: str
    score: float
    tokens: str


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)


@app.get("/")
def root():
    return {
        "project": "Projet 11 - Analyse des discours politiques",
        "author": __author__,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": service.ready,
        "author": __author__,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        return service.predict(payload.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Modele absent dans models/.") from exc


@app.post("/predict/batch", response_model=List[PredictResponse])
def predict_batch(payload: BatchRequest):
    try:
        return [service.predict(text) for text in payload.texts]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Modele absent dans models/.") from exc
