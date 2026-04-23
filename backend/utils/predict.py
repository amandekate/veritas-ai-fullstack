import numpy as np
import requests
from typing import Any


TEXT_API_URL = "https://amandekate-veritas-text-api.hf.space/predict"


def _clip_probability(value: float) -> float:
   
    return float(np.clip(value, 0.0, 1.0))


def predict_text(headline: str) -> float:
   
    try:
        response = requests.post(
            TEXT_API_URL,
            json={"text": headline},
            timeout=15
        )

        if response.status_code != 200:
            print("Text API failed:", response.text)
            return 0.5

        data = response.json()

        return _clip_probability(float(data.get("score", 0.5)))

    except Exception as e:
        print("Text API error:", e)
        return 0.5


def predict_image(image_array: np.ndarray, model: Any) -> float:
    """
    Predict fake probability using image model
    """
    raw_prediction = model.predict(image_array, verbose=0)

    prediction = np.asarray(raw_prediction).squeeze()

    if prediction.ndim == 0:
        score = float(prediction)

    elif prediction.ndim == 1:
        if prediction.size == 1:
            score = float(prediction[0])
        elif prediction.size >= 2:
            score = float(prediction[1])
        else:
            score = float(prediction[0])

    else:
        score = float(prediction.reshape(-1)[0])

    if score < 0.0 or score > 1.0:
        score = 1 / (1 + np.exp(-score))

    return _clip_probability(score)


def fuse_predictions(
    p_text: float,
    p_image: float,
    w_text: float = 0.6,
    w_image: float = 0.4,
) -> tuple[str, float]:
    """
    Combine text + image predictions using weighted average
    """

    p_text = _clip_probability(p_text)
    p_image = _clip_probability(p_image)

    total_weight = w_text + w_image
    if total_weight <= 0:
        raise ValueError("Weights must be positive")

    fake_probability = ((p_text * w_text) + (p_image * w_image)) / total_weight
    fake_probability = _clip_probability(fake_probability)

    if fake_probability >= 0.5:
        return "FAKE", fake_probability

    return "REAL", 1.0 - fake_probability