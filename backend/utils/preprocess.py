from io import BytesIO
from typing import Any

import numpy as np
from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.resnet50 import preprocess_input


def preprocess_text(headline: str, tokenizer: Any | None = None):
    cleaned = headline.strip()
    if not cleaned:
        raise ValueError("Headline cannot be empty.")

    if tokenizer:
        return tokenizer(cleaned, truncation=True, padding=True, return_tensors="tf")

    return cleaned


async def preprocess_image(file: UploadFile) -> np.ndarray:
    if not file.content_type.startswith("image/"):
        raise ValueError("Uploaded file must be an image.")

    try:
        contents = await file.read()
        if not contents:
            raise ValueError("Image is empty.")

        with Image.open(BytesIO(contents)) as image:
            image = image.convert("RGB")

            image = image.resize((160, 160))

            image_array = np.asarray(image, dtype=np.float32)

    except UnidentifiedImageError:
        raise ValueError("Invalid image file.")
    except Exception as exc:
        raise ValueError(f"Image processing error: {exc}")

    image_array = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_array)