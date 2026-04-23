import os
from pathlib import Path
from typing import Any

import requests
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "final_image_model.keras"

_image_model: Any | None = None


class ModelLoadError(RuntimeError):
    pass


def download_model(url: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print("Downloading model from Hugging Face...")

        response = requests.get(url, stream=True)

        if response.status_code != 200:
            raise ModelLoadError(f"Download failed: {response.status_code}")

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(8192):
                if chunk:
                    f.write(chunk)

    except Exception as e:
        raise ModelLoadError(f"Download error: {e}")

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ModelLoadError("Downloaded file is empty")

    print("Download complete")


def load_image_model():
    global _image_model

    if _image_model is not None:
        return _image_model

    MODEL_URL = "https://huggingface.co/amandekate/veritas-image-model/resolve/main/final_image_model.keras"

    if not MODEL_PATH.exists():
        download_model(MODEL_URL, MODEL_PATH)

    try:
        print("Loading model...")
        _image_model = keras.models.load_model(str(MODEL_PATH))
        print("Model loaded")
    except Exception as e:
        raise ModelLoadError(f"Load failed: {e}")

    return _image_model