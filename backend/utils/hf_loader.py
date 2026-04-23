from pathlib import Path
from typing import Any
import requests
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "final_image_model.keras"

_model: Any | None = None


class ModelLoadError(RuntimeError):
    pass


def download_model(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise ModelLoadError(f"Download failed: {r.status_code}")

    with open(path, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)


def load_image_model():
    global _model

    if _model is not None:
        return _model

    url = "https://huggingface.co/amandekate/veritas-image-model/resolve/main/final_image_model.keras"

    if not MODEL_PATH.exists():
        print("Downloading from HF...")
        download_model(url, MODEL_PATH)

    print("Loading model...")
    _model = keras.models.load_model(str(MODEL_PATH))
    print("Model loaded")

    return _model