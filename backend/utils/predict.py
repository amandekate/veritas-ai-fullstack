import numpy as np
from typing import Any


def _clip_probability(value: float) -> float:
    """
    Ensure probability stays between 0 and 1
    """
    return float(np.clip(value, 0.0, 1.0))


def predict_text(headline: str) -> float:
    """
    Mock text prediction (since BERT model is not loaded)

    Returns probability of FAKE news
    """
    headline = headline.lower()

    if "breaking" in headline or "shocking" in headline:
        return 0.8
    elif "official" in headline or "confirmed" in headline:
        return 0.2
    elif "fake" in headline or "rumor" in headline:
        return 0.9

    return 0.5


def predict_image(image_array: np.ndarray, model: Any) -> float:
    """
    Predict fake probability using image model
    """
    raw_prediction = model.predict(image_array, verbose=0)

    prediction = np.asarray(raw_prediction).squeeze()

    # Handle different output shapes
    if prediction.ndim == 0:
        score = float(prediction)

    elif prediction.ndim == 1:
        if prediction.size == 1:
            score = float(prediction[0])
        elif prediction.size >= 2:
            # Assuming index 1 = FAKE class
            score = float(prediction[1])
        else:
            score = float(prediction[0])

    else:
        score = float(prediction.reshape(-1)[0])

    # Convert logits to probability if needed
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