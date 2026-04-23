from contextlib import asynccontextmanager
from typing import Any
import logging
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from utils.hf_loader import ModelLoadError, load_image_model
from utils.predict import fuse_predictions, predict_image, predict_text
from utils.preprocess import preprocess_image, preprocess_text


logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.image_model = None
    app.state.model_error = None

    try:
        app.state.image_model = load_image_model()
        logging.info("Image model loaded successfully")
    except ModelLoadError as exc:
        app.state.model_error = str(exc)
        logging.error(f"Model load error: {exc}")

    yield


app = FastAPI(
    title="Multimodal Fake News Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home() -> dict[str, str | bool | None]:
    image_model = getattr(app.state, "image_model", None)
    model_error = getattr(app.state, "model_error", None)

    return {
        "message": "Backend is running",
        "image_model_loaded": image_model is not None,
        "model_error": model_error,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    headline: str = Form(...),
    file: UploadFile = File(...),
    w_text: float = Form(0.6),
    w_image: float = Form(0.4),
) -> dict[str, str | float]:

    logging.info(f"Received headline: {headline}")
    logging.info(f"Weights -> text: {w_text}, image: {w_image}")

    image_model: Any | None = getattr(app.state, "image_model", None)
    if image_model is None:
        detail = getattr(app.state, "model_error", None) or "Image model is not loaded."
        raise HTTPException(status_code=503, detail=detail)

    try:
        processed_headline = preprocess_text(headline)
        image_array = await preprocess_image(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        start_time = time.time()

        text_score = predict_text(processed_headline)
        image_score = predict_image(image_array, image_model)

        
        label, confidence = fuse_predictions(
            text_score,
            image_score,
            w_text=w_text,
            w_image=w_image
        )

        end_time = time.time()
        logging.info(f"Prediction took {end_time - start_time:.2f} seconds")

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "text_score": round(text_score, 4),
        "image_score": round(image_score, 4),
        "status": "success",
    }